//! Admission du constructeur BVH complet (VIctory #2118).

use std::collections::HashMap;
use std::sync::{Arc, Mutex, Once, OnceLock};

use super::{Shader, make};

struct Builder {
    members: HashMap<blake3::Hash, Arc<make::Pending>>,
    start: Once,
    decision: Mutex<Option<Decision>>,
}

#[derive(Clone, Copy)]
enum Decision {
    Preparing(u64),
    Final(bool),
}

fn decide(last: &mut Option<Decision>, frame: u64, status: impl FnOnce() -> Option<bool>) -> bool {
    match *last {
        Some(Decision::Final(ready)) => return ready,
        Some(Decision::Preparing(f)) if f == frame => return false,
        _ => {}
    }
    let ready = status();
    if ready.is_none() && last.is_none() {
        tracing::info!(frame, "BVH builder held: preparing pipelines");
    }
    *last = Some(ready.map_or(Decision::Preparing(frame), Decision::Final));
    ready == Some(true)
}

static BUILDER: OnceLock<Builder> = OnceLock::new();

/// Enregistre les conteneurs de la bibliothèque avant l'appel à DllMain.
pub fn register_builder<'a>(containers: impl Iterator<Item = &'a [u8]>) -> anyhow::Result<()> {
    let mut members = HashMap::new();
    for bytes in containers {
        members.entry(blake3::hash(bytes)).or_insert_with(|| {
            let shader = super::shader("compute", bytes);
            Arc::new(make::Pending::new(vec![shader], None, None))
        });
    }
    anyhow::ensure!(!members.is_empty(), "empty BVH builder inventory");
    let count = members.len();
    if let Err(other) = BUILDER.set(Builder {
        members,
        start: Once::new(),
        decision: Mutex::new(None),
    }) {
        let first = BUILDER.get().unwrap();
        anyhow::ensure!(
            first.members.len() == count
                && other
                    .members
                    .keys()
                    .all(|key| first.members.contains_key(key)),
            "BVH builder inventory changed during the session"
        );
    } else {
        tracing::info!(members = count, "BVH builder inventory");
    }
    Ok(())
}

pub(super) fn member(shaders: &[Arc<Shader>]) -> Option<Arc<make::Pending>> {
    let [cs] = shaders else { return None };
    if cs.stage != "compute" {
        return None;
    }
    let group = BUILDER.get()?;
    let host = group.members.get(&blake3::hash(&cs.container))?;
    // Une copie du conteneur et une création via un flux de sous-objets
    // retrouvent le même pipeline hôte, y compris après un Release du titre.
    group.start.call_once(|| {
        for member in group.members.values() {
            make::start_builder(member);
        }
    });
    Some(host.clone())
}

thread_local! {
    /// The verdict of the builder call this thread is in ([`builder_call`]).
    static CALL: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

/// A call into the title's BVH builder, which records every stage of its
/// builds or copies into the title's list on the calling thread. Until it
/// is dropped, each of those dispatches takes the verdict the call had at
/// its start, so the call runs whole or not at all (VIctory #2118). A
/// verdict read again at each dispatch held a call's first stages and ran
/// the next ones once the frame advanced and the pipelines were ready.
#[must_use = "the verdict holds while the call lives"]
pub struct BuilderCall {
    /// The verdict of the call this one is made in, given back after it.
    outer: Option<bool>,
    admitted: bool,
}

impl BuilderCall {
    /// Whether the call's dispatches run.
    pub fn admitted(&self) -> bool {
        self.admitted
    }
}

impl Drop for BuilderCall {
    fn drop(&mut self) {
        CALL.set(self.outer);
    }
}

/// Starts a call into the title's BVH builder on this thread: see
/// [`BuilderCall`]. A call made inside another takes the outer verdict.
pub fn builder_call() -> BuilderCall {
    enter(|| BUILDER.get().is_none() || decided())
}

fn enter(verdict: impl FnOnce() -> bool) -> BuilderCall {
    let outer = CALL.get();
    let admitted = outer.unwrap_or_else(verdict);
    CALL.set(Some(admitted));
    BuilderCall { outer, admitted }
}

/// Whether the builder's dispatches run: the verdict of the builder call
/// this thread is in, else the session's at this frame.
pub(super) fn admitted() -> bool {
    CALL.get().unwrap_or_else(decided)
}

/// Whether a BVH builder inventory is registered and currently held (not admitted).
pub(super) fn held() -> bool {
    BUILDER.get().is_some() && !admitted()
}

fn decided() -> bool {
    let group = BUILDER.get().expect("registered builder member");
    let mut last = group.decision.lock().unwrap_or_else(|e| e.into_inner());
    decide(&mut last, crate::frame::current(), || {
        #[cfg(feature = "diagnostics")]
        if group.members.keys().any(|h| {
            super::skip_hashes()
                .iter()
                .any(|s| &h.to_hex()[..16] == s.as_str())
        }) {
            tracing::info!(
                members = group.members.len(),
                "BVH builder held: member explicitly skipped"
            );
            return Some(false);
        }
        make::admit_builder(group.members.values())
    })
}

#[test]
fn builder_finishes_preparation_between_frames() {
    let mut decision = None;
    assert!(!decide(&mut decision, 3, || None));
    assert!(!decide(&mut decision, 3, || panic!("même image")));
    assert!(decide(&mut decision, 4, || Some(true)));
    assert!(decide(&mut decision, 5, || panic!("verdict définitif")));
    let mut refused = None;
    assert!(!decide(&mut refused, 3, || Some(false)));
    assert!(!decide(&mut refused, 4, || panic!("refus définitif")));
}

/// Every dispatch of one builder call takes the verdict of its start, even
/// if the session's changes meanwhile; a call inside it takes it too.
#[test]
fn builder_call_keeps_its_verdict_for_every_stage() {
    let held = enter(|| false);
    assert!(!held.admitted());
    assert!(!admitted(), "the frozen verdict, not the session's");
    {
        let inner = enter(|| panic!("an inner call takes the outer verdict"));
        assert!(!inner.admitted());
    }
    assert_eq!(CALL.get(), Some(false));
    drop(held);
    assert_eq!(CALL.get(), None);
    let run = enter(|| true);
    assert!(run.admitted() && admitted());
    drop(run);
    assert_eq!(CALL.get(), None);
}
