use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

use crate::{Console, Unimplemented};

#[cfg(windows)]
struct RunTarget<'a> {
    exe: PathBuf,
    package_dir: &'a Path,
    title: xtp_hle::runtime::Title,
}

#[cfg(windows)]
fn resolve_target(path: &Path) -> Result<RunTarget<'_>> {
    let folder = path.is_dir();
    let package_dir = if folder {
        path
    } else {
        path.parent().unwrap_or(Path::new("."))
    };
    let manifest_path = package_dir.join("MicrosoftGame.config");
    let manifest = match std::fs::read_to_string(&manifest_path) {
        Ok(text) => text,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => String::new(),
        Err(e) => {
            eprintln!("warning: could not read {}: {e}", manifest_path.display());
            String::new()
        }
    };
    let title = xtp_hle::runtime::Title::from_manifest(&manifest);
    let exe = if folder {
        pick_executable(package_dir, &title.executables)?
    } else {
        path.to_owned()
    };
    Ok(RunTarget {
        exe,
        package_dir,
        title,
    })
}

#[cfg(windows)]
fn configured_executable(root: &Path, name: &str) -> Result<PathBuf> {
    use std::path::Component;

    let path = Path::new(name);
    anyhow::ensure!(is_exe(path), "expected an .exe extension");
    anyhow::ensure!(
        path.components()
            .all(|c| matches!(c, Component::Normal(_) | Component::CurDir)),
        "expected a path inside the package"
    );
    let file = root.join(path).canonicalize()?;
    let relative = file
        .strip_prefix(root)
        .context("resolves outside the package")?;
    anyhow::ensure!(file.is_file(), "not a regular file");
    Ok(relative.to_owned())
}

#[cfg(windows)]
fn is_exe(path: &Path) -> bool {
    path.extension()
        .is_some_and(|ext| ext.eq_ignore_ascii_case("exe"))
}

#[cfg(windows)]
fn pick_executable(folder: &Path, configured: &[String]) -> Result<PathBuf> {
    let root = folder
        .canonicalize()
        .with_context(|| format!("opening {}", folder.display()))?;
    let mut candidates = Vec::new();
    let mut skipped = Vec::new();
    for name in configured {
        match configured_executable(&root, name) {
            Ok(path) => candidates.push(path),
            Err(e) => skipped.push(format!("{name:?}: {e}")),
        }
    }
    if candidates.is_empty() {
        for entry in std::fs::read_dir(folder)
            .with_context(|| format!("listing executables in {}", folder.display()))?
        {
            let entry = entry?;
            if is_exe(&entry.path()) && entry.file_type()?.is_file() {
                candidates.push(PathBuf::from(entry.file_name()));
            }
        }
        if !skipped.is_empty() {
            let details = skipped.join("\n  ");
            if candidates.is_empty() {
                anyhow::bail!(
                    "no executable found in {}. MicrosoftGame.config entries skipped:\n  {details}",
                    folder.display()
                );
            }
            eprintln!(
                "warning: using the top-level .exe search in {}. MicrosoftGame.config entries skipped:\n  {details}",
                folder.display()
            );
        }
    }
    candidates.sort();
    candidates.dedup();
    match candidates.as_slice() {
        [exe] => Ok(folder.join(exe)),
        [] => anyhow::bail!("no executable found in {}", folder.display()),
        _ => {
            let names = candidates
                .iter()
                .map(|p| p.display().to_string())
                .collect::<Vec<_>>();
            anyhow::bail!(
                "multiple executables in {}. Pass one to xtp run:\n  {}",
                folder.display(),
                names.join("\n  ")
            )
        }
    }
}

#[cfg(windows)]
pub fn run(
    target: &Path,
    unimplemented: Unimplemented,
    strict: bool,
    execute: bool,
    console: Console,
    autopilot: Option<PathBuf>,
) -> Result<()> {
    use std::sync::Arc;
    use xtp_hle::{HleResolver, HostPolicy};
    use xtp_loader::{BindReport, UnimplementedPolicy, WithTitleDlls};

    if execute {
        crate::preflight::check_cpu()?;
    }

    // A title that relaunched itself: the old process must be gone first.
    xtp_hle::modules::lifecycle::wait_for_previous();
    let cfg = xtp_core::Config::load()?;
    let input = &cfg.input;
    xtp_hle::modules::gameinput::configure(&xtp_hle::modules::gameinput::Settings {
        keyboard_mouse: input.keyboard_mouse,
        mouse_sensitivity: input.mouse_sensitivity,
        mouse_curve: input.mouse_curve,
        mouse_dead_zone: input.mouse_dead_zone,
        invert_y: input.invert_y,
        bindings: input.bindings.clone(),
    })
    .map_err(|e| anyhow::anyhow!("[input] in the config: {e}"))?;
    let RunTarget {
        exe,
        package_dir,
        title,
    } = resolve_target(target)?;
    let original = std::fs::read(&exe).with_context(|| format!("reading {}", exe.display()))?;
    let prepared = xtp_loader::prepare(&original)?;
    let hash = prepared.original_hash.to_hex();
    let build = xtp_game::identify(&hash);
    if execute {
        xtp_game::ambient::configure(&hash);
        xtp_hle::modules::xmem::reserve_low_va_from_env(
            build.map_or(0, |build| build.low_va_reserve_mib),
        );
    }
    match build {
        Some(build) => println!("build          {} ({hash})", build.label),
        None => println!("build          unknown ({hash})"),
    }

    if execute && xtp_game::identify(&hash).is_some() {
        xtp_gpu::configure_copy_back(
            cfg.cache_subdir("readmap")
                .ok()
                .map(|dir| dir.join(format!("{hash}.v1.map"))),
        );
    }
    let images = cfg.cache_subdir("images")?;
    xtp_hle::modules::d3d12x::use_shader_cache(cfg.cache_subdir("shaders")?);
    let image = xtp_loader::load(&prepared, &images)?;
    // SAFETY: just mapped from the executable `hash` names; no title code has run.
    unsafe { xtp_game::watchdog::apply_if_asked(&hash, image.base(), image.info().size_of_image) };
    // SAFETY: as above.
    unsafe { xtp_game::asserts::apply_if_asked(&hash, image.base(), image.info().size_of_image) };
    // SAFETY: as above; the image stays mapped for the process.
    unsafe { xtp_game::rt_path::apply_if_asked(&hash, image.base(), image.info().size_of_image) };
    xtp_hle::runtime::set_main_image(image.base());
    xtp_hle::runtime::set_image_path(image.base(), &exe);
    println!(
        "mapped at      {:#x} (preferred {:#x})",
        image.base(),
        image.info().image_base
    );
    let resolver = HleResolver::new(if strict {
        HostPolicy::none()
    } else {
        HostPolicy::default()
    });
    let wanted = |dll: &str| title_dll(&resolver.host, dll);
    let title_dir = exe.parent().unwrap_or(Path::new("."));
    let storage = cfg.data_subdir(&format!("{:08X}/storage", title.id))?;
    println!(
        "title          {:08X}, storage {} ({} MB)",
        title.id,
        storage.display(),
        title.storage_bytes >> 20
    );
    // The console's fixed drives: the package as `G:`, temporary files `T:`.
    xtp_hle::modules::files::mount_at('G', package_dir);
    let temp = cfg.data_subdir(&format!("{:08X}/temp", title.id))?;
    xtp_hle::modules::files::mount_at('T', &temp);
    xtp_hle::runtime::set_title(xtp_hle::runtime::Title { storage, ..title });
    xtp_hle::console::set(match console {
        Console::SeriesX => xtp_hle::console::Console::SeriesX,
        Console::SeriesS => xtp_hle::console::Console::SeriesS,
    });
    println!("console        {}", xtp_hle::console::current().name());
    let dlls = xtp_loader::load_title_dlls(image.info(), title_dir, &images, &wanted)?;
    for (name, dll) in &dlls {
        register_builder(name, title_dir)?;
        println!("title dll      {name} at {:#x}", dll.base());
        xtp_hle::runtime::set_image_path(dll.base(), &title_dir.join(name));
    }

    let policy = match unimplemented {
        Unimplemented::Trap => UnimplementedPolicy::Trap,
        Unimplemented::Zero => UnimplementedPolicy::ReturnZero,
    };
    let with_dlls = WithTitleDlls {
        inner: &resolver,
        dlls: &dlls,
    };
    let with_dlls = xtp_hle::modules::bvhbuilder::WithBuilderTrace(&with_dlls);
    let show = |name: &str, report: BindReport| {
        println!(
            "bound          {name}: {} hle, {} host, {} title, {} missing",
            report.hle,
            report.host,
            report.title,
            report.missing.len()
        );
        for m in &report.missing {
            println!("  missing      {}!{}", m.dll, m.symbol);
        }
    };
    for (name, dll) in &dlls {
        show(name, dll.bind_imports(&with_dlls, policy)?);
    }
    let exe_name = exe.file_name().unwrap_or_default().to_string_lossy();
    show(&exe_name, image.bind_imports(&with_dlls, policy)?);
    // SAFETY: bound just above; no title code has run.
    unsafe { xtp_game::intro::apply_if_asked(&hash, image.base(), image.info().size_of_image) };
    if !execute {
        println!("not executing (pass --execute to start the title)");
        return Ok(());
    }
    xtp_loader::report_crashes();
    xtp_hle::modules::lifecycle::install_crash_restart(autopilot.is_some());
    xtp_loader::log_crashes_to(temp.with_file_name("crashes.log"));
    xtp_hle::runtime::use_console_timers();
    #[cfg(feature = "diagnostics")]
    diagnostics(&image, &cfg);
    let late = LateDlls {
        host: resolver.host.clone(),
        title_dir: title_dir.to_owned(),
        cache_dir: images,
        policy,
        mapped: std::sync::Mutex::new(dlls.clone()),
        lock: LoaderLock::default(),
    };
    xtp_hle::runtime::set_title_dll_loader(Box::new(move |file| {
        late.load(file, show).map_err(|e| format!("{e:#}"))
    }));
    if let Some(script) = autopilot {
        xtp_hle::modules::xgameui::set_auto_dismiss(true);
        let size = image.info().size_of_image as usize;
        xtp_hle::modules::autopilot::start(&script, &hash, image.base(), size)
            .map_err(anyhow::Error::msg)?;
    }
    // SAFETY: imports are bound; from here on we run title code.
    let main = unsafe { Arc::new(image).spawn_main(dlls)? };
    let code = main
        .join()
        .map_err(|_| anyhow::anyhow!("title main thread panicked"))??;
    xtp_gpu::save_copy_back_map();
    println!("title exited with {code}");
    Ok(())
}

/// The debugging aids (README): scheduled stack dumps, watched values and
/// memory saved on a crash.
#[cfg(all(windows, feature = "diagnostics"))]
fn diagnostics(image: &xtp_loader::LoadedImage, cfg: &xtp_core::Config) {
    xtp_loader::dump_stacks_on_schedule();
    xtp_hle::modules::audio::capture::start();
    watch(image.base(), image.info().size_of_image as usize);
    // Debugging aid: `XTP_CRASH_SAVE=<rva>:<len>,...` (hex) writes those
    // title ranges to `crash-<rva>.bin` in the cache dir on a crash.
    let size = image.info().size_of_image as usize;
    let ranges = std::env::var("XTP_CRASH_SAVE").unwrap_or_default();
    let hex = |s: &str| usize::from_str_radix(s.trim().trim_start_matches("0x"), 16).ok();
    xtp_loader::save_on_crash(
        ranges
            .split(',')
            .filter_map(|r| r.split_once(':'))
            .filter_map(|(rva, len)| Some((hex(rva)?, hex(len)?)))
            .filter(|&(rva, len)| rva.checked_add(len).is_some_and(|end| end <= size))
            .map(|(rva, len)| {
                let file = cfg.cache_dir.join(format!("crash-{rva:x}.bin"));
                (image.base() + rva, len, file)
            })
            .collect(),
    );
}

/// Debugging aid: `XTP_WATCH=<rva>,...` (hex) logs the u32 at each
/// title+rva whenever it changes, polled every 5 ms (an engine's state, say).
#[cfg(all(windows, feature = "diagnostics"))]
fn watch(base: usize, size: usize) {
    let Ok(list) = std::env::var("XTP_WATCH") else {
        return;
    };
    let rvas: Vec<usize> = list
        .split(',')
        .filter_map(|r| usize::from_str_radix(r.trim().trim_start_matches("0x"), 16).ok())
        .filter(|&rva| rva + 4 <= size)
        .collect();
    std::thread::spawn(move || {
        let mut seen = vec![None; rvas.len()];
        loop {
            for (&rva, seen) in rvas.iter().zip(&mut seen) {
                // SAFETY: four bytes inside the mapped image, checked above.
                let value = unsafe { ((base + rva) as *const u32).read_volatile() };
                if *seen != Some(value) {
                    eprintln!("XTP: title+{rva:#x} = {value:#x}");
                    *seen = Some(value);
                }
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
    });
}

#[cfg(windows)]
fn register_builder(name: &str, dir: &Path) -> Result<()> {
    if name.eq_ignore_ascii_case("XboxBvhBuilder.dll") {
        let path = dir.join(name);
        let bytes = std::fs::read(&path).with_context(|| format!("reading {}", path.display()))?;
        // The survey and the game path walk the same validated containers.
        let containers = crate::shader::sources(&bytes).into_iter().filter_map(|s| {
            let crate::shader::Source::Raw { data, .. } = s else {
                return None;
            };
            let size = u32::from_le_bytes(data[24..28].try_into().unwrap()) as usize;
            Some(&data[..size])
        });
        xtp_gpu::pipeline::register_builder(containers)?;
    }
    Ok(())
}

/// DLLs the host or our runtime provide are never taken from the title.
pub(crate) fn title_dll(host: &xtp_hle::HostPolicy, dll: &str) -> bool {
    !host.allows(dll) && !xtp_hle::registry().iter().any(|m| m.serves(dll))
}

/// Title DLLs the title loads itself (`LoadLibrary`) that nothing imports,
/// e.g. BTP 6's upscaler behind its FidelityFX loader.
#[cfg(windows)]
struct LateDlls {
    host: xtp_hle::HostPolicy,
    title_dir: std::path::PathBuf,
    cache_dir: std::path::PathBuf,
    policy: xtp_loader::UnimplementedPolicy,
    /// Every title DLL mapped so far, to bind new ones against.
    mapped: std::sync::Mutex<Vec<(String, xtp_loader::LoadedImage)>>,
    lock: LoaderLock,
}

#[cfg(windows)]
impl LateDlls {
    /// Every `LoadLibrary` of the title comes here first. Returns a title
    /// image called `file`, mapping it and the title DLLs it needs first,
    /// binding them and running their `DllMain` on the calling thread, as
    /// the Windows loader would; `None` if `file` is no title image.
    fn load(
        &self,
        file: &str,
        show: impl Fn(&str, xtp_loader::BindReport),
    ) -> Result<Option<usize>> {
        use xtp_hle::runtime::{forget_image, image_named, set_image_path};
        // Waits while another thread initialises a DLL; a `DllMain` that
        // loads another DLL re-enters.
        self.lock.hold(|| {
            if let Some(base) = image_named(file) {
                return Ok(Some(base));
            }
            let new = {
                let mut mapped = self.mapped.lock().unwrap_or_else(|e| e.into_inner());
                let wanted = |dll: &str| title_dll(&self.host, dll) && image_named(dll).is_none();
                let new =
                    xtp_loader::load_title_dll(file, &self.title_dir, &self.cache_dir, &wanted)?;
                let hle = xtp_hle::HleResolver::new(self.host.clone());
                let old = xtp_loader::WithTitleDlls {
                    inner: &hle,
                    dlls: &mapped,
                };
                let resolver = xtp_loader::WithTitleDlls {
                    inner: &old,
                    dlls: &new,
                };
                let resolver = xtp_hle::modules::bvhbuilder::WithBuilderTrace(&resolver);
                for (name, dll) in &new {
                    register_builder(name, &self.title_dir)?;
                    println!(
                        "title dll      {name} at {:#x} (loaded by the title)",
                        dll.base()
                    );
                    show(name, dll.bind_imports(&resolver, self.policy)?);
                    set_image_path(dll.base(), &self.title_dir.join(name));
                }
                // Visible before their `DllMain`s run, which may load more.
                mapped.extend(new.iter().cloned());
                new
            };
            let failed = new.iter().enumerate().find_map(|(i, (name, dll))| {
                // SAFETY: bound above, dependencies attached first.
                let attached = unsafe { dll.attach() };
                attached
                    .err()
                    .map(|e| (i, e.context(format!("attaching {name}"))))
            });
            let Some((i, error)) = failed else {
                return Ok(new.last().map(|(_, dll)| dll.base()));
            };
            // As `LoadLibrary` does when a `DllMain` fails: detach what this
            // load attached (the failed one too) and unmap all of it.
            let bases: Vec<_> = new.iter().map(|(_, d)| d.base()).collect();
            self.mapped
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .retain(|(_, d)| !bases.contains(&d.base()));
            for (_, done) in new[..=i].iter().rev() {
                // SAFETY: attached (or failing to) on this thread just now.
                unsafe { done.detach() };
            }
            for (_, dll) in new {
                forget_image(dll.base());
                // SAFETY: detached, and no title code holds it yet.
                unsafe { dll.unload() };
            }
            Err(error)
        })
    }
}

/// The Windows loader lock's shape: one thread at a time, re-entered by
/// the thread that holds it.
#[cfg(windows)]
#[derive(Default)]
struct LoaderLock {
    owner: std::sync::Mutex<(Option<std::thread::ThreadId>, usize)>,
    free: std::sync::Condvar,
}

#[cfg(windows)]
impl LoaderLock {
    fn hold<T>(&self, f: impl FnOnce() -> T) -> T {
        let me = std::thread::current().id();
        let lock = || self.owner.lock().unwrap_or_else(|e| e.into_inner());
        {
            let mut owner = lock();
            while owner.0.is_some_and(|o| o != me) {
                owner = self.free.wait(owner).unwrap_or_else(|e| e.into_inner());
            }
            *owner = (Some(me), owner.1 + 1);
        }
        let result = f();
        let mut owner = lock();
        owner.1 -= 1;
        if owner.1 == 0 {
            owner.0 = None;
            self.free.notify_one();
        }
        result
    }
}

#[cfg(all(test, windows))]
mod tests {
    use std::fs;
    use std::sync::{Arc, Mutex};

    use super::resolve_target;

    #[test]
    fn folder_prefers_the_configured_executable() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        fs::create_dir(root.join("bin")).unwrap();
        let selected = root.join("bin").join("Title & Tools.exe");
        fs::write(&selected, []).unwrap();
        fs::write(root.join("Helper.exe"), []).unwrap();
        fs::write(
            root.join("MicrosoftGame.config"),
            r#"<Game><TitleId>2A</TitleId>
              <!-- <ExecutableList><Executable Name="Helper.exe"/></ExecutableList> -->
              <Other><Executable Name="Helper.exe"/></Other>
              <ExecutableList><Executable DisplayName="A > B" Name = 'bin/Title &amp; Tools.exe'/>
              </ExecutableList></Game>"#,
        )
        .unwrap();
        let target = resolve_target(root).unwrap();
        assert_eq!(target.exe, selected);
        assert_eq!(target.package_dir, root);
        assert_eq!(target.title.id, 42);
    }

    #[test]
    fn folder_falls_back_to_its_only_executable() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        let exe = root.join("Only.EXE");
        fs::write(&exe, []).unwrap();
        fs::create_dir(root.join("not-a-file.exe")).unwrap();
        assert_eq!(resolve_target(root).unwrap().exe, exe);
        fs::write(
            root.join("MicrosoftGame.config"),
            r#"<Game><ExecutableList><Executable Name="missing.exe"/></ExecutableList></Game>"#,
        )
        .unwrap();
        assert_eq!(resolve_target(root).unwrap().exe, exe);
    }

    #[test]
    fn ambiguous_folder_lists_candidates_and_accepts_an_explicit_path() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        for name in ["B.exe", "A.exe"] {
            fs::write(root.join(name), []).unwrap();
        }
        let error = resolve_target(root).err().unwrap().to_string();
        assert!(error.ends_with("\n  A.exe\n  B.exe"), "{error}");
        let exe = root.join("B.exe");
        assert_eq!(resolve_target(&exe).unwrap().exe, exe);
    }

    #[test]
    fn several_configured_executables_list_only_the_matches() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        for name in ["Tool.exe", "A.exe", "B.exe"] {
            fs::write(root.join(name), []).unwrap();
        }
        fs::write(
            root.join("MicrosoftGame.config"),
            r#"<Game><ExecutableList><Executable Name="B.exe"/><Executable Name="A.exe"/>
              <Executable Name="B.exe"/></ExecutableList></Game>"#,
        )
        .unwrap();
        let error = resolve_target(root).err().unwrap().to_string();
        assert!(error.ends_with("\n  A.exe\n  B.exe"), "{error}");
    }

    #[test]
    fn empty_folder_does_not_follow_a_manifest_outside_the_package() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("package");
        fs::create_dir(&root).unwrap();
        fs::write(dir.path().join("Outside.exe"), []).unwrap();
        fs::write(
            root.join("MicrosoftGame.config"),
            r#"<Game><ExecutableList><Executable Name="../Outside.exe"/><Executable Name="readme.txt"/>
                <Executable Name="missing.exe"/></ExecutableList></Game>"#,
        )
        .unwrap();
        let error = resolve_target(&root).err().unwrap().to_string();
        assert!(error.contains("no executable found"), "{error}");
        for reason in [
            "Outside.exe",
            "inside the package",
            "readme.txt",
            ".exe extension",
            "missing.exe",
        ] {
            assert!(error.contains(reason), "{error}");
        }
        assert!(error.contains(root.to_str().unwrap()), "{error}");
    }

    #[test]
    fn loader_lock_is_reentrant_and_exclusive() {
        let lock = Arc::new(super::LoaderLock::default());
        assert_eq!(lock.hold(|| lock.hold(|| 7)), 7);
        let order = Arc::new(Mutex::new(Vec::new()));
        let other = lock.hold(|| {
            let (theirs, their_order) = (lock.clone(), order.clone());
            let other = std::thread::spawn(move || {
                theirs.hold(|| their_order.lock().unwrap().push(2));
            });
            std::thread::sleep(std::time::Duration::from_millis(50));
            order.lock().unwrap().push(1);
            other
        });
        other.join().unwrap();
        assert_eq!(*order.lock().unwrap(), [1, 2]);
    }
}

#[cfg(not(windows))]
pub fn run(
    _: &Path,
    _: Unimplemented,
    _: bool,
    _: bool,
    _: Console,
    _: Option<PathBuf>,
) -> Result<()> {
    anyhow::bail!("running titles requires Windows")
}
