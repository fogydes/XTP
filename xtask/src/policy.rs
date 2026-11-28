//! Rule #1, statically: the repository only ever contains small text files
//! that we wrote. See docs/asset-policy.md.
//!
//! This is an allowlist, not a blocklist: a file passes only if its name is
//! on the list, it is not too big, it is text, and it does not smuggle data
//! inside text (long hex/base64 runs or decimal byte arrays).

use std::path::Path;
use std::process::Command;

use anyhow::{Context, Result, bail};

const MAX_FILE_SIZE: usize = 512 * 1024;
const MAX_ENCODED_RUN: usize = 512;
// The XMA decoder has a 272-entry decimal VLC table.
const MAX_DECIMAL_BYTES: usize = 512;

const ALLOWED_EXTENSIONS: &[&str] = &[
    "rs", "toml", "md", "txt", "lock", "yml", "yaml", "json", "hlsl", "hlsli", "ps1", "cmd", "sh",
];
const ALLOWED_NAMES: &[&str] = &[
    ".gitignore",
    ".gitattributes",
    ".editorconfig",
    ".xtp-root",
    "pre-commit",
    "commit-msg",
    "CODEOWNERS",
    "LICENSE",
    "LICENSE-MIT",
    "LICENSE-APACHE",
];

/// Paths explicitly exempted, with a reason. Keep this empty if at all possible.
const EXCEPTIONS: &[(&str, &str)] = &[(
    "crates/xtp-formats/src/bcpack-tables.bin",
    "the BCPack format's constant tables: nothing decodes without them and they cannot be derived (owner's decision; docs/asset-policy.md)",
)];

/// Commercial titles and their publishers are not named in this repository
/// (CONTRIBUTING.md, "Writing about titles"). Neither are a title's setting
/// or a local game folder. Stored reversed so that this file does not
/// contain them; matched case-insensitively, in every file.
const FORBIDDEN_NAMES_REVERSED: &[&str] = &[
    // a title
    "otua tfeht dnarg",
    "iv atg",
    "6 atg",
    "6atg",
    "6-atg",
    "iv-atg",
    "6_atg",
    "iv_atg",
    "ivatg",
    // its publisher and the publisher's parent company
    "ratskcor",
    "owt-ekat",
    "owtekat",
    "owt_ekat",
    "evitcaretni owt ekat",
    // its setting: the state and the city
    "adinoel",
    "ytic eciv",
    "yticeciv",
    "ytic-eciv",
    "ytic_eciv",
    // a local game folder
    "uwu\\:e",
    "uwu/:e",
];

/// A title's file names, stored reversed like the names above. They may
/// appear only under `TITLE_FILE_PATHS`, where that title's build identity,
/// hooks, launch defaults and autopilot script live.
const TITLE_FILE_NAMES_REVERSED: &[&str] = &["retsam_ttelracs_emag"];
const TITLE_FILE_PATHS: &[&str] = &["crates/xtp-game/", "xtask/", "autopilot/gameplay.txt"];

/// The names of `reversed` found in `text` (already lower-case).
fn found(text: &str, reversed: &[&str]) -> usize {
    reversed
        .iter()
        .map(|r| r.chars().rev().collect::<String>())
        .filter(|name| text.contains(name.as_str()))
        .count()
}

/// Signatures of formats that must never be committed.
const FORBIDDEN_MAGIC: &[(&[u8], &str)] = &[
    (b"MZ", "PE executable"),
    (b"DXBC", "shader container"),
    (b"DDS ", "DDS texture"),
    (b"RIFF", "RIFF (audio/video)"),
    (b"BC\xC0\xDE", "LLVM bitcode"),
    (b"Microsoft C/C++ MSF", "PDB"),
    (b"!<arch>", "library archive"),
    (b"PK\x03\x04", "zip"),
    (b"7z\xBC\xAF", "7-zip"),
    (b"MSCF", "cabinet"),
    (b"\xD0\xCF\x11\xE0", "MSI/OLE"),
];

pub fn run(root: &Path, staged: bool) -> Result<()> {
    let files = list_files(root, staged)?;
    let mut violations = Vec::new();
    for path in &files {
        if EXCEPTIONS.iter().any(|(p, _)| p == path) {
            continue;
        }
        let content = if staged {
            let out = Command::new("git")
                .args(["show", &format!(":{path}")])
                .current_dir(root)
                .output()?;
            if !out.status.success() {
                continue; // deleted in the index
            }
            out.stdout
        } else {
            match std::fs::read(root.join(path)) {
                Ok(c) => c,
                Err(_) => continue,
            }
        };
        for problem in check(path, &content) {
            violations.push(format!("{path}: {problem}"));
        }
    }
    if violations.is_empty() {
        println!("asset policy: {} files ok", files.len());
        return Ok(());
    }
    for v in &violations {
        eprintln!("POLICY: {v}");
    }
    bail!(
        "{} asset-policy violation(s). Nothing from the game may be committed; \
         see docs/asset-policy.md",
        violations.len()
    )
}

fn list_files(root: &Path, staged: bool) -> Result<Vec<String>> {
    let git = |args: &[&str]| -> Result<Option<Vec<String>>> {
        let out = Command::new("git")
            .args(args)
            .current_dir(root)
            .output()
            .context("running git")?;
        if !out.status.success() {
            return Ok(None);
        }
        Ok(Some(
            out.stdout
                .split(|&b| b == 0)
                .filter(|s| !s.is_empty())
                .map(|s| String::from_utf8_lossy(s).replace('\\', "/"))
                .collect(),
        ))
    };
    if staged {
        return git(&[
            "diff",
            "--cached",
            "--name-only",
            "--diff-filter=ACMR",
            "-z",
        ])?
        .context("--staged needs a git repository");
    }
    if let Some(mut tracked) = git(&[
        "ls-files",
        "-z",
        "--cached",
        "--others",
        "--exclude-standard",
    ])? {
        tracked.sort();
        tracked.dedup();
        return Ok(tracked);
    }
    // Not a git repository: walk everything except build output.
    Ok(walkdir::WalkDir::new(root)
        .into_iter()
        .filter_entry(|e| !matches!(e.file_name().to_str(), Some("target" | ".git")))
        .filter_map(Result::ok)
        .filter(|e| e.file_type().is_file())
        .filter_map(|e| {
            e.path()
                .strip_prefix(root)
                .ok()
                .map(|p| p.to_string_lossy().replace('\\', "/"))
        })
        .collect())
}

/// All problems with one file. Pure, for testing.
pub fn check(path: &str, content: &[u8]) -> Vec<String> {
    let mut problems = Vec::new();
    let name = path.rsplit('/').next().unwrap_or(path);
    let ext = name.rsplit_once('.').map(|(_, e)| e.to_ascii_lowercase());
    let allowed = ALLOWED_NAMES.contains(&name)
        || ext
            .as_deref()
            .is_some_and(|e| ALLOWED_EXTENSIONS.contains(&e));
    if !allowed {
        problems.push("file type not on the allowlist".to_owned());
    }
    if content.len() > MAX_FILE_SIZE {
        problems.push(format!(
            "{} bytes exceeds the {MAX_FILE_SIZE}-byte limit",
            content.len()
        ));
    }
    for (magic, what) in FORBIDDEN_MAGIC {
        if content.starts_with(magic) {
            problems.push(format!("looks like a {what}"));
        }
    }
    if content.contains(&0) {
        problems.push("binary content (NUL bytes)".to_owned());
    } else {
        if let Some(run) = longest_encoded_run(content)
            && run > MAX_ENCODED_RUN
        {
            problems.push(format!("{run}-character hex/base64 run (embedded data?)"));
        }
        let decimal_bytes = longest_decimal_byte_array(content);
        if decimal_bytes > MAX_DECIMAL_BYTES {
            problems.push(format!(
                "{decimal_bytes} decimal bytes in one array (embedded data?)"
            ));
        }
    }
    if !content.contains(&0) {
        let text = String::from_utf8_lossy(content).to_lowercase();
        let names = found(&text, FORBIDDEN_NAMES_REVERSED);
        if names > 0 {
            problems.push(format!(
                "names a commercial title, its publisher or its setting, or a local game \
                 folder ({names} match); describe the platform behaviour instead \
                 (CONTRIBUTING.md, \"Writing about titles\")"
            ));
        }
        if !TITLE_FILE_PATHS.iter().any(|p| path.starts_with(p)) {
            let files = found(&text, TITLE_FILE_NAMES_REVERSED);
            if files > 0 {
                problems.push(format!(
                    "names a title's file ({files} match); outside {} say \"the title \
                     executable\"",
                    TITLE_FILE_PATHS.join(", ")
                ));
            }
        }
    }
    problems
}

/// Longest run of characters that could be hex or base64 payload.
fn longest_encoded_run(content: &[u8]) -> Option<usize> {
    let is_payload = |b: &u8| b.is_ascii_alphanumeric() || matches!(b, b'+' | b'/' | b'=');
    content.split(|b| !is_payload(b)).map(<[u8]>::len).max()
}

/// Recognize bracketed, comma-separated decimal bytes in source or JSON.
fn longest_decimal_byte_array(content: &[u8]) -> usize {
    content
        .split(|&byte| byte == b'[')
        .skip(1)
        .filter_map(|tail| {
            let close = tail.iter().position(|&byte| byte == b']')?;
            let array = tail[..close].trim_ascii();
            let values = array.strip_suffix(b",").unwrap_or(array);
            let mut count = 0;
            for value in values.split(|&byte| byte == b',') {
                let value = value.trim_ascii();
                let digits = value
                    .strip_suffix(b"_u8")
                    .or_else(|| value.strip_suffix(b"u8"))
                    .unwrap_or(value);
                if !digits.iter().all(u8::is_ascii_digit)
                    || std::str::from_utf8(digits).ok()?.parse::<u8>().is_err()
                {
                    return None;
                }
                count += 1;
            }
            Some(count)
        })
        .max()
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::{MAX_DECIMAL_BYTES, check};

    /// Enforces the policy wherever tests run, even in clones without the hook.
    #[test]
    fn repository_passes_policy() {
        super::run(&crate::workspace_root(), false).unwrap();
    }

    #[test]
    fn accepts_source() {
        assert!(check("crates/x/src/lib.rs", b"fn main() {}\n").is_empty());
        assert!(check(".gitignore", b"/target/\n").is_empty());
        assert!(check(".githooks/pre-commit", b"#!/bin/sh\n").is_empty());
        assert!(check(".githooks/commit-msg", b"#!/bin/sh\n").is_empty());
        assert!(check(".github/CODEOWNERS", b"# owners\n").is_empty());
    }

    #[test]
    fn rejects_game_like_files() {
        assert!(!check("assets/tex.dds", b"DDS \x7c\0\0\0").is_empty());
        assert!(
            !check("notes.txt", b"MZ\x90\0").is_empty(),
            "magic wins over extension"
        );
        assert!(
            !check("data.bin", b"hello").is_empty(),
            "extension not allowed"
        );
        assert!(
            !check("a.rs", &vec![b'a'; 600 * 1024]).is_empty(),
            "too big"
        );
    }

    fn unreversed(reversed: &str) -> String {
        reversed.chars().rev().collect()
    }

    #[test]
    fn rejects_the_title_name() {
        let name = unreversed("otua tfeht dnarg");
        let text = format!("// as {} does\n", name.to_uppercase());
        assert!(!check("crates/x/src/lib.rs", text.as_bytes()).is_empty());
        assert!(check("crates/x/src/lib.rs", b"// as BTP 6 does\n").is_empty());
        assert!(!check("docs/analysis/notes.md", text.as_bytes()).is_empty());
    }

    #[test]
    fn rejects_the_parent_company_the_setting_and_the_game_folder() {
        for reversed in [
            "evitcaretni owt ekat",
            "owt-ekat",
            "adinoel",
            "ytic eciv",
            "uwu\\:e",
        ] {
            let name = unreversed(reversed);
            for text in [
                format!("seen in {name}\n"),
                format!("SEEN IN {}\n", name.to_uppercase()),
            ] {
                for path in ["docs/analysis/notes.md", "crates/xtp-game/src/lib.rs"] {
                    assert!(
                        !check(path, text.as_bytes()).is_empty(),
                        "{reversed} in {path}"
                    );
                }
            }
        }
        assert!(check("docs/a.md", b"read `<title folder>\\shaders.rpf`\n").is_empty());
        assert!(
            check("crates/x/src/lib.rs", b"// take two arguments\n").is_empty(),
            "plain English stays allowed"
        );
    }

    #[test]
    fn names_the_title_executable_only_where_its_build_lives() {
        let text = format!("run `{}.exe`\n", unreversed("retsam_ttelracs_emag"));
        for path in [
            "crates/xtp-game/src/lib.rs",
            "xtask/src/play.rs",
            "autopilot/gameplay.txt",
        ] {
            assert!(check(path, text.as_bytes()).is_empty(), "{path}");
        }
        for path in [
            "docs/dev/autopilot.md",
            "apps/xtp/src/main.rs",
            "autopilot/other.txt",
            "crates/xtp-gpu/src/lib.rs",
        ] {
            assert!(!check(path, text.as_bytes()).is_empty(), "{path}");
        }
    }

    #[test]
    fn rejects_data_smuggled_into_text() {
        let blob = format!("const DATA: &str = \"{}\";", "A1".repeat(400));
        assert!(!check("src/blob.rs", blob.as_bytes()).is_empty());
    }

    #[test]
    fn rejects_data_smuggled_as_a_decimal_byte_array() {
        let values: Vec<String> = (0..=MAX_DECIMAL_BYTES)
            .map(|i| ((i * 73 + 17) % 256).to_string())
            .collect();
        let short = format!("[{}]", values[..MAX_DECIMAL_BYTES].join(", "));
        assert!(check("table.json", short.as_bytes()).is_empty());

        let data = format!("const DATA: &[u8] = &[{}u8,\n];\n", values.join("_u8, "));
        assert_eq!(
            check("src/data.rs", data.as_bytes()),
            [format!(
                "{} decimal bytes in one array (embedded data?)",
                values.len()
            )]
        );

        let statement = format!("[{}]; [0, 1]", values[..MAX_DECIMAL_BYTES].join(", "));
        assert!(check("src/table.rs", statement.as_bytes()).is_empty());
    }
}
