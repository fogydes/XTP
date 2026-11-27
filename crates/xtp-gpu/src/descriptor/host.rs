//! Host descriptor heaps behind the title's shader-visible heaps (decision
//! 0007, step 3). Translated shaders index the heaps directly at the
//! title's descriptor positions, so each host heap keeps the title heap's
//! order: slot `i` is made from what the title wrote at slot `i`, when it
//! writes it (view creation and descriptor copies) and again when a draw
//! reads it through a table and finds the slot changed.

use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock, Weak};

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::core::Interface;
use xtp_shader::emit::{RawWindow, StrideEntry};

use super::{VIEW_TAG, View, view_at};
use crate::host::host;
use crate::shadow::Shadow;

/// Logs `what` the first time it happens.
macro_rules! once {
    ($($arg:tt)*) => {{
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| tracing::warn!($($arg)*));
    }};
}

pub(crate) struct HostHeap {
    pub(crate) heap: ID3D12DescriptorHeap,
    cpu: usize,
    step: u32,
    /// Slots mirroring the title's; view heaps have `dynamic` more.
    count: u32,
    /// Dynamic slots past `count` (none in sampler heaps): [`DYNAMIC`]
    /// where the host takes them on top of every title slot or caps the
    /// heap ([`capped_split`]), else [`MIN_DYNAMIC`].
    dynamic: u32,
    sampler: bool,
    /// The record (view index and generation) each slot was last made from,
    /// and for a texture the title placed, the host texture.
    made: Mutex<Made>,
    /// The host resource each slot views, held while the slot does: shaders
    /// index heaps directly, so a slot the title has not rewritten since it
    /// released the resource may still be read, and a freed host resource
    /// read there is a GPU page fault (the console reads stale memory).
    kept: Mutex<Vec<Option<ID3D12Resource>>>,
    /// Free ranges between descriptors held by recordings and shared views.
    slots: Mutex<Slots>,
    /// Raw views of host copies that every list naming the same range
    /// shares (root SRVs), in chunks of dynamic slots. Locked before
    /// `slots` when both are.
    shared: Mutex<Shared>,
    /// A raw view that reads zeros, in a dynamic slot of its own (made on
    /// first use, kept for the heap's life).
    null_raw: Mutex<Option<u32>>,
    /// The runtime stride table and the raw views it names, in a view heap
    /// of title slots with `XTP_STRIDE_TABLE=1`
    /// ([`crate::device::stride_table_on`]); shared with [`FENCED`], which
    /// gives retired raw views back when the GPU fence reaches them.
    strides: Option<Arc<Strides>>,
    /// The sweeps of [`HostHeap::picked_spans`], and the listing they
    /// return (its [`Made::listing`]).
    picked: Mutex<Sweeps>,
    listed: Arc<Picked>,
    /// View heaps: the title slots holding a view of a texture the title
    /// may fill, for shaders that pick textures by index
    /// ([`TextureQueue`], [`HostHeap::texture_pass`]).
    textures: Mutex<TextureQueue>,
    /// View heaps: per title slot, the frame (plus one; 0: never) in which
    /// a draw last read it through a table
    /// ([`HostHeap::texture_slot_tabled`]).
    tabled: Vec<std::sync::atomic::AtomicU32>,
    /// View heaps: passes over the textures picked by index taken in a
    /// frame that had one already (frames that do not advance), each a
    /// frame more on the queue's clock ([`HostHeap::clock`]).
    passes_on: AtomicU64,
}

/// Slots past the dynamic ones in a view heap with a runtime stride table,
/// for the raw views its entries name ([`Strides`]): one per window of
/// host copy that the heap's structured or typed buffer views read (slots
/// whose views read the same bytes share one), and one per such window no
/// view reads any more while a submission that may read it has not run.
/// One per slot ran out about 30 s into the world (image-status.md §5.5:
/// 65,536 of 65,536 with none retiring, 78,000 to 83,000 title slots
/// holding buffer views, and the tables made afresh copying them into
/// dynamic slots), and reads through the table gave zeros from then on.
/// shortcut: a guess (VIctory #704's), not a measured need; past it a view
/// gets no raw view, so its entry holds none and reads through the table
/// give zeros until a raw view is free again (logged).
const ALIASES: u32 = 1 << 16;

/// Entries before slot 0's in the table ([`xtp_shader::emit::STRIDE_HEADER`]).
const HEADER: u32 = xtp_shader::emit::STRIDE_HEADER;

/// A view heap's runtime stride table: an entry per slot
/// ([`StrideEntry`]), written with the slot's descriptor, and the raw
/// views the entries name. Shaders that read a buffer by element without
/// a stride known at translation read the entry of the slot they compute,
/// then access the raw view it names (VIctory #193 and #704).
///
/// Every write of an entry, and every change to the raw views, happens
/// under the pool's lock ([`Strides::pool`]), so a raw view handed to a
/// slot whose view had none ([`Aliases::refused_slots`]) never lands after
/// the slot changed.
struct Strides {
    table: StrideTable,
    /// The first of the pool's slots, past the dynamic ones.
    alias_base: u32,
    /// The title slots the heap mirrors: slots from here to `alias_base`
    /// are dynamic ones.
    mirrored: u32,
    /// The heap the raw views are made in, held only to keep `cpu` valid:
    /// a fence's completion may make one null after the heap's owner let
    /// go. Then its first CPU handle and its step.
    _heap: ID3D12DescriptorHeap,
    cpu: usize,
    step: u32,
    aliases: Mutex<Aliases>,
}

// SAFETY: D3D12 descriptor heaps and resources are free-threaded; the
// table's mapping is written with atomic stores only; the rest is behind
// the pool's mutex.
unsafe impl Send for Strides {}
// SAFETY: as above.
unsafe impl Sync for Strides {}

/// The stride tables whose pool has raw views waiting for the host fence
/// ([`Aliases::retiring`]): each completion of a submission gives back
/// those it reached ([`fence_reached`]), whether or not anything touches
/// the heap again (Codex review of 244cf78, finding 3: they came back only
/// at the heap's next use). Locked before a pool's lock, never after.
static FENCED: Mutex<Vec<Weak<Strides>>> = Mutex::new(Vec::new());

/// Gives back the retired raw views of every stride table that the host
/// fence has reached ([`crate::host::Host`] calls it each time it finishes
/// the submissions the GPU has run): made null, back in their pool, their
/// host buffers dropped; and the dynamic slots whose view the full pool
/// gave none get one ([`Aliases::refused_slots`]).
pub(crate) fn fence_reached() {
    let mut fenced = FENCED.lock().unwrap_or_else(|e| e.into_inner());
    if fenced.is_empty() {
        return;
    }
    let mut live = Vec::with_capacity(fenced.len());
    let mut gone = Vec::new();
    fenced.retain(|weak| {
        let Some(s) = weak.upgrade() else {
            return false;
        };
        let mut a = s.pool();
        s.reap(&mut a);
        gone.append(&mut a.gone);
        let waits = !a.retiring.is_empty();
        a.listed = waits;
        drop(a);
        live.push(s);
        waits
    });
    drop(fenced);
    // A last reference may free video memory, which can wait.
    drop(gone);
    drop(live);
}

/// An upload buffer of one 8-byte entry per heap slot after a header,
/// mapped for good: the header says how many entries follow and how many
/// title slots the heap mirrors, for shaders to read only entries inside
/// it ([`xtp_shader::emit::STRIDE_HEADER`]; the root SRV that binds it has
/// no size).
/// shortcut: shaders read it from system memory over PCIe (VIctory #704's
/// testers measured frame times 7 to 11 % higher with the table); copy it
/// to the card per submission if a session shows the reads cost.
struct StrideTable {
    resource: ID3D12Resource,
    /// The mapping's address.
    ptr: usize,
    gpu: u64,
    /// Its entries: every slot of the heap.
    len: u32,
}

impl StrideTable {
    /// A table of `len` entries of zeros (no view in any slot) for a heap
    /// mirroring `mirrored` title slots.
    fn new(device: &ID3D12Device, len: u32, mirrored: u32) -> Option<StrideTable> {
        let bytes = u64::from(len + HEADER) * u64::from(StrideEntry::BYTES);
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: bytes,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        };
        let mut resource: Option<ID3D12Resource> = None;
        let mut p = std::ptr::null_mut();
        // SAFETY: a plain upload buffer, mapped for good; the zeros and the
        // header are written inside the mapping.
        let made = unsafe {
            device
                .CreateCommittedResource(
                    &D3D12_HEAP_PROPERTIES {
                        Type: D3D12_HEAP_TYPE_UPLOAD,
                        ..Default::default()
                    },
                    D3D12_HEAP_FLAG_NONE,
                    &desc,
                    D3D12_RESOURCE_STATE_GENERIC_READ,
                    None,
                    &mut resource,
                )
                .and_then(|()| {
                    let r = resource.as_ref().ok_or_else(windows::core::Error::empty)?;
                    r.Map(0, None, Some(&mut p))?;
                    std::ptr::write_bytes(p.cast::<u8>(), 0, bytes as usize);
                    let header = p.cast::<u64>();
                    header.write(u64::from(mirrored.min(len)) << 32);
                    header.add(1).write(u64::from(len) << 32);
                    Ok(r.GetGPUVirtualAddress())
                })
        };
        match (made, resource) {
            (Ok(gpu), Some(resource)) => Some(StrideTable {
                resource,
                ptr: p as usize,
                gpu,
                len,
            }),
            (made, _) => {
                tracing::warn!(error = ?made.err(), len, "no runtime stride table: shaders that read one are dropped");
                None
            }
        }
    }
}

/// The raw views of a heap's stride table, in a pool of slots from
/// [`Strides::alias_base`] ([`ALIASES`] of them).
///
/// A raw view is made once per window of host copy (the buffer, its first
/// dword, its dwords, unordered access or not) and named by every slot
/// whose view reads that window ([`Aliases::named`]): a view the title
/// copies into many slots, or that the tables made afresh copy into
/// dynamic slots, takes one pool slot, not one per slot. Its bytes never
/// change while a slot names it.
///
/// A slot whose view changes window names another raw view (made before
/// its entry names it, if no slot named that window yet), and the old one
/// retires once no slot names it: a submission the GPU may be running read
/// the old entry (stride, bias, count) and then reads the old raw view,
/// which must stay as it was (rewriting it in place, as before, gave such
/// a submission the new window with the old stride and bias). A retired
/// pool slot comes back once the host fence has reached the submission
/// that was handed to the queue last when it retired
/// ([`crate::host::Host::executing`]): later submissions read the table as
/// it is after, so none of them reads it. It is made null then, and the
/// host buffer it viewed goes: when the host finishes that submission
/// ([`fence_reached`]), or sooner if the heap is used meanwhile. A
/// recording that keeps a list for many submissions holds nothing (holds
/// by recording pinned the pool for good before: stride-2, 65,536 taken,
/// 49,829 retiring), so the retiring ones are those of the submissions in
/// flight, at most the pool.
#[derive(Default)]
struct Aliases {
    /// The window of the raw view each slot that has one names.
    of: HashMap<u32, RawKey>,
    /// The raw views named, by window: each named by the slots of `of`
    /// that hold its window.
    named: HashMap<RawKey, Alias>,
    /// Pool slots free now (null views).
    free: Vec<u32>,
    /// Pool slots no entry names any more, oldest first: the fence value
    /// they wait for, the pool slot, whether it is an unordered view, and
    /// the host buffer it viewed.
    retiring: VecDeque<(u64, u32, bool, ID3D12Resource)>,
    /// Pool slots from here on were never taken.
    next: u32,
    /// The pool's slots ([`ALIASES`]; fewer in tests).
    size: u32,
    /// Pool slots taken (named or retiring), and the most at once.
    used: u32,
    peak: u32,
    /// Raw views a slot needed that the full pool did not give, since the
    /// start.
    refused: u64,
    /// The dynamic slots whose buffer view the full pool gave no raw view
    /// (their entry says none), and what that view reads: each gets one
    /// as soon as one comes back ([`Strides::reap`]). A table made afresh
    /// is kept for the list's later draws with the same descriptors
    /// (`draw.rs` `fresh_tables`), whose shaders read the entry when they
    /// run: it names the view then (Codex review of 244cf78, finding 4:
    /// the entry stayed none for good). Title slots are made again by the
    /// next draw or sweep that reads them instead. With the window each
    /// waits for, and the same by window: a raw view of it made for another
    /// slot is theirs too at once, at no cost, the pool full or not (Codex
    /// review of 8612e27, a new bug: the slot waited for a pool slot to come
    /// back, and `reap` stopped once the pool was full again, though the
    /// first one back had just made the window it waited for).
    refused_slots: BTreeMap<u32, (Viewed, RawKey)>,
    waiting_for: HashMap<RawKey, BTreeSet<u32>>,
    /// The table is in [`FENCED`], or about to be (`to_list`).
    listed: bool,
    to_list: bool,
    /// Host buffers of raw views given back, dropped once the lock is let
    /// go ([`Strides::with_pool`]).
    gone: Vec<ID3D12Resource>,
}

/// What a raw view views: its host buffer (by address, held by its
/// [`Alias`]), first dword, dwords, and whether it is for unordered access.
type RawKey = (usize, u64, u32, bool);

/// A raw view of the pool: its pool slot, the host buffer it views, and the
/// slots naming it.
struct Alias {
    pool: u32,
    buffer: ID3D12Resource,
    slots: u32,
}

/// What [`HostHeap::note_view`] did with a slot's entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Noted {
    /// The heap has no table, or the slot holds no buffer view it names.
    Nothing,
    /// The entry names a raw view of the view's bytes.
    Named,
    /// The view needs a raw view the full pool did not give: none for now,
    /// the slot is to be made again ([`Aliases`] gives one back once the
    /// GPU is done with it).
    Refused,
}

/// The host fence value a pool slot that retires now waits for, read after
/// the entry no longer names it ([`crate::host::Host::executing`]); 0
/// without a host (nothing runs).
fn retire_ticket() -> u64 {
    std::sync::atomic::fence(Ordering::SeqCst);
    host().map_or(0, crate::host::Host::executing)
}

/// The host fence value the GPU has reached; everything without a host.
fn completed() -> u64 {
    host().map_or(u64::MAX, crate::host::Host::completed)
}

impl Aliases {
    /// A free pool slot, or none if every one is named or retiring.
    fn take(&mut self) -> Option<u32> {
        let pool = self.free.pop().or_else(|| {
            (self.next < self.size).then(|| {
                self.next += 1;
                self.next - 1
            })
        });
        if pool.is_none() {
            self.refused += 1;
            POOL_REFUSED.fetch_add(1, Ordering::Relaxed);
            return None;
        }
        self.used += 1;
        if self.used > self.peak {
            POOL_PEAK.fetch_max(self.used, Ordering::Relaxed);
            // Logged at each new eighth of the pool.
            let eighth = |v: u32| u64::from(v) * 8 / u64::from(self.size.max(1));
            if eighth(self.used) > eighth(self.peak) {
                tracing::info!(
                    peak = self.used,
                    of = self.size,
                    retiring = self.retiring.len(),
                    named = self.named.len(),
                    naming_slots = self.of.len(),
                    "runtime stride table raw views: a new peak in use"
                );
            }
            self.peak = self.used;
        }
        pool
    }

    /// Whether a pool slot is free now ([`Aliases::take`] would give one).
    fn has_free(&self) -> bool {
        !self.free.is_empty() || self.next < self.size
    }

    /// Slot `index` no longer names the raw view of window `key`, which
    /// retires if no other slot names it: given back once the GPU has
    /// reached the fence value [`retire_ticket`] reads now (after the
    /// slot's entry changed); the table goes in [`FENCED`] if it is not
    /// there ([`Strides::with_pool`]).
    fn unname(&mut self, key: RawKey) {
        let Some(alias) = self.named.get_mut(&key) else {
            return;
        };
        alias.slots -= 1;
        if alias.slots > 0 {
            return;
        }
        let Some(alias) = self.named.remove(&key) else {
            return;
        };
        self.retiring
            .push_back((retire_ticket(), alias.pool, key.3, alias.buffer));
        if !self.listed {
            self.listed = true;
            self.to_list = true;
        }
    }

    /// The retiring pool slots the GPU is done with at `completed`, out of
    /// the queue: each to be made null, then free ([`HostHeap::reap`]).
    fn done(&mut self, completed: u64) -> Vec<(u32, bool, ID3D12Resource)> {
        let mut done = Vec::new();
        while self
            .retiring
            .front()
            .is_some_and(|&(ticket, ..)| ticket <= completed)
        {
            if let Some((_, pool, unordered, buffer)) = self.retiring.pop_front() {
                done.push((pool, unordered, buffer));
            }
        }
        done
    }

    /// Dynamic slot `index`, whose view `v` reads window `key`, waits for a
    /// raw view ([`Aliases::refused_slots`]).
    fn wait(&mut self, index: u32, v: &Viewed, key: RawKey) {
        self.unwait(index);
        self.refused_slots.insert(index, (v.clone(), key));
        self.waiting_for.entry(key).or_default().insert(index);
    }

    /// Slot `index` waits for a raw view no more: its view, if it did.
    fn unwait(&mut self, index: u32) -> Option<Viewed> {
        let (v, key) = self.refused_slots.remove(&index)?;
        if let Some(slots) = self.waiting_for.get_mut(&key) {
            slots.remove(&index);
            if slots.is_empty() {
                self.waiting_for.remove(&key);
            }
        }
        Some(v)
    }
}

impl Strides {
    /// The pool, locked.
    fn pool(&self) -> std::sync::MutexGuard<'_, Aliases> {
        self.aliases.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Runs `f` with the pool locked; then, the lock let go, drops the
    /// host buffers of the raw views it gave back and, if a raw view
    /// retired into an empty queue, lists the table in [`FENCED`] and
    /// gives back what the fence has reached already (the submission it
    /// waits for may have finished before the table was listed, and no
    /// later completion would look).
    fn with_pool<R>(self: &Arc<Self>, f: impl FnOnce(&Self, &mut Aliases) -> R) -> R {
        let mut a = self.pool();
        let r = f(self, &mut a);
        let gone = std::mem::take(&mut a.gone);
        let listed = std::mem::take(&mut a.to_list);
        drop(a);
        drop(gone);
        if listed {
            FENCED
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .push(Arc::downgrade(self));
            let mut a = self.pool();
            self.reap(&mut a);
            let gone = std::mem::take(&mut a.gone);
            drop(a);
            drop(gone);
        }
        r
    }

    /// The CPU handle of heap slot `index`.
    fn cpu(&self, index: u32) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        D3D12_CPU_DESCRIPTOR_HANDLE {
            ptr: self.cpu + index as usize * self.step as usize,
        }
    }

    /// Whether `index` is one of the heap's dynamic slots.
    fn dynamic(&self, index: u32) -> bool {
        (self.mirrored..self.alias_base).contains(&index)
    }

    /// Writes slot `index`'s table entry, in one 64-bit store: a shader
    /// reading it (one load of its two dwords) never sees half of an old
    /// entry and half of a new one.
    fn set_entry(&self, index: u32, entry: StrideEntry) {
        if index >= self.table.len {
            return;
        }
        let bits = entry.bits().unwrap_or_else(|| {
            once!(?entry, "a stride table entry the table cannot hold: none");
            0
        });
        // SAFETY: an 8-byte aligned entry inside the live mapping, past
        // the header.
        let at = unsafe {
            AtomicU64::from_ptr((self.table.ptr as *mut u64).add((index + HEADER) as usize))
        };
        at.store(bits, Ordering::Release);
    }

    /// Gives the retiring raw views the GPU is done with back to the pool
    /// ([`Strides::reclaim`]), then hands free ones to the dynamic slots
    /// the full pool gave none ([`Aliases::refused_slots`]), oldest slot
    /// first.
    fn reap(&self, a: &mut Aliases) {
        self.reclaim(a);
        while a.has_free() {
            let Some((&index, &(_, key))) = a.refused_slots.first_key_value() else {
                break;
            };
            let Some(v) = a.unwait(index) else {
                break;
            };
            // Slots waiting for the same window get the raw view made for
            // it as well ([`Strides::name`]).
            if self.name(a, index, &v) == Noted::Refused {
                a.wait(index, &v, key);
                break;
            }
        }
    }

    /// Gives the retiring raw views the GPU is done with back to the pool,
    /// each made null first: no submission reads it any more, and a stale
    /// read of the slot would otherwise find a view of a buffer that went.
    /// Their host buffers go to [`Aliases::gone`], dropped after the lock
    /// (a last reference may free video memory, which can wait).
    fn reclaim(&self, a: &mut Aliases) {
        if a.retiring.is_empty() {
            return;
        }
        for (pool, unordered, buffer) in a.done(completed()) {
            let at = self.cpu(self.alias_base + pool);
            if unordered {
                null_raw_uav(at);
            } else {
                null_buffer_view(at);
            }
            a.free.push(pool);
            a.used -= 1;
            a.gone.push(buffer);
        }
    }

    /// Slot `index`'s table entry for the view just made there: `viewed`,
    /// the bytes of its host copy a buffer view reads (none for any other
    /// view, or one the host could not make), named as [`Strides::name`]
    /// does. A dynamic slot whose view the full pool gives no raw view
    /// waits for one ([`Aliases::refused_slots`]); any other change of the
    /// slot ends that wait.
    fn note(self: &Arc<Self>, index: u32, viewed: Option<&Viewed>) -> Noted {
        self.with_pool(|s, a| {
            a.unwait(index);
            let noted = match viewed {
                Some(v) => s.name(a, index, v),
                None => {
                    s.unname(a, index);
                    Noted::Nothing
                }
            };
            if noted == Noted::Refused
                && s.dynamic(index)
                && let Some(v) = viewed
                && let Some((_, _, key)) = v.window()
            {
                a.wait(index, v, key);
            }
            s.reap(a);
            noted
        })
    }

    /// Slot `index`'s entry for view `v`, read through raw view `pool` of
    /// its window `window` of elements of `element` bytes.
    fn entry(&self, v: &Viewed, window: RawWindow, element: u32, pool: u32) -> StrideEntry {
        StrideEntry {
            stride: element,
            raw_slot: Some(self.alias_base + pool),
            bias: window.bias,
            count: (v.len / v.element).min(u64::from(window.elements(element))) as u32,
        }
    }

    /// Slot `index`'s entry for buffer view `v`. A raw view is read in its
    /// own slot; a structured or typed one through a raw view of the same
    /// bytes in the pool: the one other slots name for that window, or one
    /// made before the entry names it. A slot whose view changes window
    /// names another raw view, and the old one retires once no slot names
    /// it, until the GPU has run what may read it ([`Aliases`]); one whose
    /// window stays keeps its raw view, untouched.
    fn name(&self, a: &mut Aliases, index: u32, v: &Viewed) -> Noted {
        if v.raw {
            self.set_entry(
                index,
                StrideEntry {
                    stride: 0,
                    raw_slot: Some(index),
                    bias: 0,
                    count: u32::try_from(v.len / 4).unwrap_or(u32::MAX),
                },
            );
            self.unalias(a, index);
            return Noted::Named;
        }
        let Some((window, element, key)) = v.window() else {
            // shortcut: a view starting off a dword of its host copy, or of
            // elements no entry holds, is read through the table as no view.
            once!(
                start = v.start,
                element = v.element,
                "a buffer view the stride table cannot name a raw view for: read as none"
            );
            self.unname(a, index);
            return Noted::Nothing;
        };
        let buffer = &v.copy.buffer;
        let entry = |pool: u32| self.entry(v, window, element, pool);
        if a.of.get(&index) == Some(&key)
            && let Some(alias) = a.named.get(&key)
        {
            // The same raw view: a submission in flight that read the old
            // entry reads the same bytes through it.
            self.set_entry(index, entry(alias.pool));
            return Noted::Named;
        }
        let pool = if let Some(alias) = a.named.get_mut(&key) {
            // Another slot's view reads the same bytes: its raw view, as it
            // is (named, it never changes).
            alias.slots += 1;
            alias.pool
        } else {
            self.reclaim(a);
            let Some(pool) = a.take() else {
                if a.refused.is_power_of_two() {
                    tracing::warn!(
                        refused = a.refused,
                        of = a.size,
                        named = a.named.len(),
                        naming_slots = a.of.len(),
                        "no free raw view slots for the stride table: a buffer read through it \
                         reads zeros until one is given back"
                    );
                }
                self.unname(a, index);
                return Noted::Refused;
            };
            raw_view(
                buffer,
                window,
                v.unordered,
                self.cpu(self.alias_base + pool),
            );
            a.named.insert(
                key,
                Alias {
                    pool,
                    buffer: buffer.clone(),
                    slots: 1,
                },
            );
            self.give_waiting(a, key, pool);
            pool
        };
        let old = a.of.insert(index, key);
        self.set_entry(index, entry(pool));
        if let Some(old) = old {
            a.unname(old);
        }
        Noted::Named
    }

    /// Raw view `pool` was just made for window `key`: the dynamic slots
    /// that waited for that window name it too, the pool full or not.
    fn give_waiting(&self, a: &mut Aliases, key: RawKey, pool: u32) {
        let Some(slots) = a.waiting_for.remove(&key) else {
            return;
        };
        for index in slots {
            let Some((v, _)) = a.refused_slots.remove(&index) else {
                continue;
            };
            let Some((window, element, _)) = v.window() else {
                continue;
            };
            if let Some(alias) = a.named.get_mut(&key) {
                alias.slots += 1;
            }
            // A waiting slot names no raw view (its refusal unnamed it).
            let old = a.of.insert(index, key);
            self.set_entry(index, self.entry(&v, window, element, pool));
            if let Some(old) = old {
                a.unname(old);
            }
        }
    }

    /// Slot `index`'s entry says none, then its raw view retires if no
    /// other slot names it.
    fn unname(&self, a: &mut Aliases, index: u32) {
        self.set_entry(index, StrideEntry::NONE);
        self.unalias(a, index);
    }

    /// Slot `index` names no raw view any more (its entry no longer names
    /// one): the one it named retires if no other slot names it.
    fn unalias(&self, a: &mut Aliases, index: u32) {
        if let Some(old) = a.of.remove(&index) {
            a.unname(old);
        }
    }
}

/// A recording's note that its shaders read a view heap's stride table
/// (`draw.rs` keeps one per heap, with the table). It holds nothing: a raw
/// view goes back to the pool once the GPU has run the submissions that
/// may read it ([`Aliases`]), whichever recordings they came from.
pub(crate) struct StrideHold;

/// The bytes of its host copy a buffer view the host made reads, for the
/// raw view of the runtime stride table ([`make_with`]) and the spans
/// shaders that pick slots at run time read ([`Made::read`]).
#[derive(Clone)]
struct Viewed {
    copy: Arc<Shadow>,
    /// The title address of the view's first byte.
    title: u64,
    /// The view's first byte in the copy, and its bytes (cut to the copy).
    start: u64,
    len: u64,
    /// Bytes per element (4 for a raw view).
    element: u64,
    /// The view is raw (read in its own slot).
    raw: bool,
    unordered: bool,
}

impl Viewed {
    /// The window of its host copy a raw view of the view reads, its bytes
    /// per element, and the raw view's key; none if the stride table cannot
    /// name one ([`Strides::name`]).
    fn window(&self) -> Option<(RawWindow, u32, RawKey)> {
        let element = u32::try_from(self.element).unwrap_or(0);
        let window = RawWindow::covering(self.start, self.len).filter(|_| {
            element > 0 && element.is_multiple_of(4) && element <= StrideEntry::MAX_STRIDE
        })?;
        let key = (
            self.copy.buffer.as_raw() as usize,
            window.first,
            window.dwords,
            self.unordered,
        );
        Some((window, element, key))
    }
}

/// What each title slot of a host heap was made from.
struct Made {
    /// The record (view index and generation; 0 when not made).
    keys: Vec<u64>,
    /// For a texture view, the host texture the slot views (0: none, the
    /// slot is null) and, for a texture the title placed itself, the
    /// resource made for its memory (see [`HostHeap::mirror`]).
    textures: HashMap<u32, (usize, usize)>,
    /// Title slots whose buffer view waits for a draw (made null when the
    /// title wrote it), each once: [`HostHeap::picked_spans`] makes them.
    /// With `XTP_RUN_TIME_VIEWS=1` only ([`run_time_views`]).
    waiting: Vec<u32>,
    queued: Vec<bool>,
    /// The buffer views made in title slots whose bytes a shader reads
    /// (through the slot, or through the raw view the table names), with
    /// `XTP_RUN_TIME_VIEWS=1` only: the title memory each reads, as cut to
    /// its host copy.
    buffers: HashMap<u32, Read>,
    /// The same by host copy, for shaders that pick slots at run time, kept
    /// as the views change (locked after `made`): shared with the heap's
    /// [`Picked`].
    listing: Arc<Mutex<Listing>>,
}

/// What a buffer view of a title slot reads ([`Made::buffers`]): its host
/// copy (by address), the title memory `[from, to)` and whether it is for
/// unordered access.
#[derive(Clone, Copy)]
struct Read {
    copy: usize,
    from: u64,
    to: u64,
    unordered: bool,
}

impl Made {
    /// Title slot `index` now holds a buffer view whose bytes `viewed` are,
    /// in place of what it held.
    fn read(&mut self, index: u32, viewed: &Viewed) {
        self.unread(index);
        let read = Read {
            copy: Arc::as_ptr(&viewed.copy) as usize,
            from: viewed.title,
            to: viewed.title + viewed.len,
            unordered: viewed.unordered,
        };
        self.listing().add(&viewed.copy, &read);
        self.buffers.insert(index, read);
    }

    /// Title slot `index` holds no buffer view whose bytes a shader reads.
    fn unread(&mut self, index: u32) {
        if let Some(read) = self.buffers.remove(&index) {
            self.listing().remove(&read);
        }
    }

    fn listing(&self) -> std::sync::MutexGuard<'_, Listing> {
        self.listing.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Title slot `index`'s buffer view waits for a sweep, if it does not
    /// already.
    fn wait(&mut self, index: u32) {
        if let Some(queued) = self.queued.get_mut(index as usize)
            && !*queued
        {
            *queued = true;
            self.waiting.push(index);
        }
    }
}

/// Views of a host copy closer than this are refreshed as one run: the
/// write watch and the copies work in pages of this size
/// ([`crate::watch::PAGE`]), so a gap under it costs a page at most.
const RUN_GAP: u64 = crate::watch::PAGE as u64;

/// The title memory the buffer views of a heap's title slots read and
/// write, by host copy, for shaders that pick slots when they run
/// ([`HostHeap::picked_spans`]); kept as the views come and go
/// ([`Made::read`]), with the order the copies changed in, so that what a
/// submission gives to refresh catches up on what changed since it last
/// looked ([`Due`]): a step per host copy that changed.
///
/// Codex review of a090f22, P2: a listing was made afresh at each change,
/// a step per host copy, and a new buffer before each of N dispatches made
/// N(N+1)/2 steps. And what a copy's views read is their runs now, not the
/// span from the lowest view to the highest (image-status.md §5.1: 4,000
/// copies listed 1.9 GB, refreshed by every list that picks, 8 to 17 GB
/// copied a frame, much of it memory no view reads, unmapped memory read
/// as zeros among it).
///
/// One listing per view heap, shared by every list whose shaders pick its
/// slots, and walked once per submission whichever lists ask
/// ([`PickedStep`]). Ported from Marvisak/VIctory (6d021fe) by Marvisak:
/// its title memory page table is one per view heap, shared by lists, and
/// the copies it names are refreshed once per submission. Before (stride
/// round 4b), each new recording caught up on the listing from nothing and
/// added a refresh step per run: 5,435 runs by some 42 lists, 226,575
/// refresh steps a frame (`stall-stride4b-1`).
#[derive(Default)]
struct Listing {
    copies: HashMap<usize, Copied>,
    /// Changes so far; each copy whose runs changed since is in `changed`
    /// at its last change (`changed_at`).
    changes: u64,
    changed: BTreeMap<u64, usize>,
    changed_at: HashMap<usize, u64>,
    /// Host copies looked at catching up (tests, the sweep's log).
    visits: u64,
    /// What submissions were given of it.
    due: Due,
}

/// What the submissions were given of a heap's [`Listing`]
/// ([`Listing::due`]): the runs as last given, by host copy, so that a
/// submission is given every run once, and again only those that changed
/// meanwhile or, after a step of its own that wrote title memory, all of
/// them; and the same for the runs of unordered views, declared written
/// once per submission.
///
/// What the title's other threads write into a run after its refresh in a
/// submission is taken by the next submission, not by a later list of the
/// same one (as VIctory 6d021fe and `crate::watch`'s single asking per
/// range have it): on the console it would race the GPU.
#[derive(Default)]
struct Due {
    /// The submission the runs were last given in
    /// ([`crate::shadow::submission`]), and the steps of its preparation
    /// that wrote title memory before (the batch's count).
    submission: u64,
    written: u64,
    /// The runs were given to refresh in `submission`, after `written`
    /// such steps.
    current: bool,
    /// The runs of unordered views were declared written in `submission`.
    declared: bool,
    /// The listing's change caught up to, for the runs refreshed and for
    /// those declared written (0: none).
    seen: u64,
    seen_writes: u64,
    /// Per host copy (held, so that no other takes its address), its runs
    /// as last given, and those of its unordered views as last declared.
    reads: HashMap<usize, (Arc<Shadow>, BTreeMap<u64, u64>)>,
    writes: HashMap<usize, (Arc<Shadow>, BTreeMap<u64, u64>)>,
    /// What `submission` was given, for what its GPU work touches
    /// ([`Listing::touches`]): title ranges refreshed, runs declared.
    given_reads: Vec<(usize, usize)>,
    given_writes: Vec<Span>,
}

impl Due {
    /// Catches `memo` up on the runs `listing` lists now for each copy
    /// that changed after change `seen` (those of unordered views with
    /// `writes`): the runs that are new or other than those of `memo`
    /// go into `out` unless everything is due anyway (`all`).
    fn catch_up(
        listing: &mut Listing,
        memo: &mut HashMap<usize, (Arc<Shadow>, BTreeMap<u64, u64>)>,
        seen: u64,
        writes: bool,
        all: bool,
        out: &mut Vec<Span>,
    ) {
        listing.since(seen, writes, |key, copy, runs| {
            if runs.is_empty() {
                memo.remove(&key);
                return;
            }
            let (_, was) = memo
                .entry(key)
                .or_insert_with(|| (copy.clone(), BTreeMap::new()));
            if !all {
                out.extend(
                    runs.iter()
                        .filter(|&(from, to)| was.get(from) != Some(to))
                        .map(|(&from, &to)| (copy.clone(), from, to)),
                );
            }
            was.clone_from(runs);
        });
    }

    /// Every run of `memo`.
    fn every(memo: &HashMap<usize, (Arc<Shadow>, BTreeMap<u64, u64>)>) -> Vec<Span> {
        memo.values()
            .flat_map(|(copy, runs)| runs.iter().map(|(&from, &to)| (copy.clone(), from, to)))
            .collect()
    }
}

/// The views of one host copy in a [`Listing`], which keeps the copy alive
/// meanwhile (no other takes its address).
struct Copied {
    copy: Arc<Shadow>,
    reads: Runs,
    /// Those of unordered views.
    writes: Runs,
}

/// The title memory views read in one host copy: their ranges, merged into
/// runs where they overlap or lie less than [`RUN_GAP`] apart.
#[derive(Default)]
struct Runs {
    /// Each view's range, counted (views of the same range).
    views: BTreeMap<(u64, u64), u32>,
    /// The runs: first address, end.
    runs: BTreeMap<u64, u64>,
    /// A view went since `runs` was made: made again from `views` when
    /// next asked for.
    stale: bool,
}

impl Runs {
    /// One more view of `[from, to)`.
    fn add(&mut self, from: u64, to: u64) {
        *self.views.entry((from, to)).or_default() += 1;
        if self.stale {
            return;
        }
        // The runs it reaches, and those only: runs lie more than the gap
        // apart, so they are the last ones starting before its end and
        // gap, back to the first ending after its start less the gap.
        let (mut first, mut end) = (from, to);
        let near: Vec<(u64, u64)> = self
            .runs
            .range(..=to.saturating_add(RUN_GAP))
            .rev()
            .take_while(|&(_, &e)| e.saturating_add(RUN_GAP) >= from)
            .map(|(&s, &e)| (s, e))
            .collect();
        for (s, e) in near {
            self.runs.remove(&s);
            (first, end) = (first.min(s), end.max(e));
        }
        self.runs.insert(first, end);
    }

    /// One view of `[from, to)` less. The runs are made again only if it
    /// was the range's last.
    fn remove(&mut self, from: u64, to: u64) {
        if let Some(n) = self.views.get_mut(&(from, to)) {
            *n -= 1;
            if *n == 0 {
                self.views.remove(&(from, to));
                self.stale = true;
            }
        }
    }

    /// The runs, made again from the views if one went.
    fn runs(&mut self) -> &BTreeMap<u64, u64> {
        if self.stale {
            self.stale = false;
            self.runs.clear();
            // By first address: each view joins the run before it or
            // starts one, as `add` merges.
            let mut run: Option<(u64, u64)> = None;
            for &(from, to) in self.views.keys() {
                match &mut run {
                    Some((_, end)) if from <= end.saturating_add(RUN_GAP) => {
                        *end = (*end).max(to);
                    }
                    _ => {
                        if let Some((s, e)) = run.replace((from, to)) {
                            self.runs.insert(s, e);
                        }
                    }
                }
            }
            if let Some((s, e)) = run {
                self.runs.insert(s, e);
            }
        }
        &self.runs
    }
}

impl Listing {
    /// A view of `read` (in `copy`) more.
    fn add(&mut self, copy: &Arc<Shadow>, read: &Read) {
        let c = self.copies.entry(read.copy).or_insert_with(|| Copied {
            copy: copy.clone(),
            reads: Runs::default(),
            writes: Runs::default(),
        });
        c.reads.add(read.from, read.to);
        if read.unordered {
            c.writes.add(read.from, read.to);
        }
        self.changed(read.copy);
    }

    /// A view of `read` less; a copy no view reads goes.
    fn remove(&mut self, read: &Read) {
        let Some(c) = self.copies.get_mut(&read.copy) else {
            return;
        };
        c.reads.remove(read.from, read.to);
        if read.unordered {
            c.writes.remove(read.from, read.to);
        }
        if c.reads.views.is_empty() {
            // Nothing to refresh for it, nor to declare written: the
            // submissions catch up if it comes back. Not held any more.
            self.copies.remove(&read.copy);
            if let Some(at) = self.changed_at.remove(&read.copy) {
                self.changed.remove(&at);
            }
            self.due.reads.remove(&read.copy);
            self.due.writes.remove(&read.copy);
        } else {
            self.changed(read.copy);
        }
    }

    /// The runs of `copy` changed (or may have).
    fn changed(&mut self, copy: usize) {
        self.changes += 1;
        if let Some(at) = self.changed_at.insert(copy, self.changes) {
            self.changed.remove(&at);
        }
        self.changed.insert(self.changes, copy);
    }

    /// Runs `each` for every copy whose runs changed after change `since`
    /// (every copy for 0), with its address, the copy and its runs of
    /// `writes` (those of unordered views) or of every view.
    fn since(
        &mut self,
        since: u64,
        writes: bool,
        mut each: impl FnMut(usize, &Arc<Shadow>, &BTreeMap<u64, u64>),
    ) {
        let keys: Vec<usize> = if since == 0 {
            self.copies.keys().copied().collect()
        } else {
            self.changed.range(since + 1..).map(|(_, &c)| c).collect()
        };
        for key in keys {
            self.visits += 1;
            if let Some(Copied {
                copy,
                reads,
                writes: written,
            }) = self.copies.get_mut(&key)
            {
                let runs = if writes { written.runs() } else { reads.runs() };
                each(key, copy, runs);
            }
        }
    }

    /// At a list's step in `submission` ([`PickedStep::due`]), after
    /// `written` steps of its preparation that wrote title memory: the runs
    /// to refresh, and (`writing`: the list's shaders write through slots
    /// they pick) those to declare written, each sorted by copy, then
    /// address (the order the submission's batch keeps, so that each lands
    /// at its end). The submission's first step gets every run; later ones
    /// only what changed since (none if nothing did), or every run again
    /// after a step that wrote title memory. Runs are declared written once
    /// per submission, and those that changed since.
    fn due(&mut self, submission: u64, written: u64, writing: bool) -> (Vec<Span>, Vec<Span>) {
        let mut due = std::mem::take(&mut self.due);
        if due.submission != submission {
            due.submission = submission;
            due.current = false;
            due.declared = false;
            due.given_reads.clear();
            due.given_writes.clear();
        }
        let all = !due.current || due.written != written;
        let mut reads = Vec::new();
        Due::catch_up(self, &mut due.reads, due.seen, false, all, &mut reads);
        due.seen = self.changes;
        if all {
            reads = Due::every(&due.reads);
            (due.current, due.written) = (true, written);
        }
        let mut writes = Vec::new();
        if writing {
            let all = !due.declared;
            Due::catch_up(
                self,
                &mut due.writes,
                due.seen_writes,
                true,
                all,
                &mut writes,
            );
            due.seen_writes = self.changes;
            if all {
                writes = Due::every(&due.writes);
                due.declared = true;
            }
        }
        for spans in [&mut reads, &mut writes] {
            spans.sort_unstable_by_key(|(copy, from, _)| (Arc::as_ptr(copy) as usize, *from));
        }
        due.given_reads.extend(
            reads
                .iter()
                .map(|&(_, from, to)| (from as usize, to as usize)),
        );
        due.given_writes.extend(writes.iter().cloned());
        self.due = due;
        (reads, writes)
    }

    /// What `submission` was given of the listing, once (the first list of
    /// the submission naming it takes it): the title ranges refreshed, and
    /// the runs declared written.
    fn touches(&mut self, submission: u64) -> (Vec<(usize, usize)>, Vec<Span>) {
        let due = &mut self.due;
        if due.submission != submission {
            return Default::default();
        }
        (
            std::mem::take(&mut due.given_reads),
            std::mem::take(&mut due.given_writes),
        )
    }

    /// For the sweep's log: host copies, runs, their bytes, the bytes from
    /// each copy's first view to its last (what was refreshed before), and
    /// runs written.
    fn census(&mut self) -> (usize, usize, u64, u64, usize) {
        let (mut runs, mut bytes, mut hull, mut written) = (0, 0, 0, 0);
        for c in self.copies.values_mut() {
            let r = c.reads.runs();
            runs += r.len();
            bytes += r.iter().map(|(from, to)| to - from).sum::<u64>();
            if let (Some((&first, _)), Some((_, &end))) = (r.first_key_value(), r.last_key_value())
            {
                hull += end - first;
            }
            written += c.writes.runs().len();
        }
        (self.copies.len(), runs, bytes, hull, written)
    }
}

/// A span of title memory in a host copy: the copy, the first address, the
/// end.
pub(crate) type Span = (Arc<Shadow>, u64, u64);

/// What the buffer views of a heap's title slots read and write, for shaders
/// that pick slots when they run ([`HostHeap::picked_spans`]): the heap's
/// [`Listing`], which recordings catch up on ([`PickedUse`]).
pub(crate) struct Picked {
    listing: Arc<Mutex<Listing>>,
}

impl Picked {
    fn listing(&self) -> std::sync::MutexGuard<'_, Listing> {
        self.listing.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Every run its views read, or (`writes`) its unordered views write,
    /// by address (tests).
    #[cfg(test)]
    fn all(&self, writes: bool) -> Vec<Span> {
        let mut all = Vec::new();
        let mut listing = self.listing();
        let visits = listing.visits;
        listing.since(0, writes, |_, copy, runs| {
            all.extend(runs.iter().map(|(&from, &to)| (copy.clone(), from, to)));
        });
        listing.visits = visits;
        all.sort_by_key(|&(_, from, _)| from);
        all
    }

    /// [`Picked::all`] of the reads (tests).
    #[cfg(test)]
    pub(crate) fn reads(&self) -> Vec<Span> {
        self.all(false)
    }

    /// [`Picked::all`] of the writes (tests).
    #[cfg(test)]
    pub(crate) fn writes(&self) -> Vec<Span> {
        self.all(true)
    }
}

/// A list's step for the shaders of its recording that pick slots of a view
/// heap ([`PickedUse::defer`], `crate::host`'s `Step::Picked`): at each
/// submission of the list, the runs the heap's buffer views read are
/// brought up to date, and those its unordered views write declared written
/// if a shader of the recording writes through slots it picks; once per
/// submission whichever lists ask ([`Listing::due`]).
pub(crate) struct PickedStep {
    picked: Arc<Picked>,
    /// A shader of the recording writes through slots it picks, and the
    /// pipeline the list had bound then (`crate::readmap`'s writer).
    writes: std::sync::atomic::AtomicBool,
    writer: AtomicU64,
}

impl PickedStep {
    /// At the step's run in `submission`, after `written` steps of its
    /// preparation that wrote title memory: the runs to refresh and those
    /// to declare written ([`Listing::due`]).
    pub(crate) fn due(&self, submission: u64, written: u64) -> (Vec<Span>, Vec<Span>) {
        let writing = self.writes.load(Ordering::Relaxed);
        let (reads, writes) = self.picked.listing().due(submission, written, writing);
        let bytes: u64 = reads.iter().map(|(_, from, to)| to - from).sum();
        for (counter, n) in [
            (0, reads.len() as u64),
            (1, bytes),
            (2, writes.len() as u64),
        ] {
            PICKS[counter].fetch_add(n, Ordering::Relaxed);
            PICKED_FRAME[counter].fetch_add(n, Ordering::Relaxed);
        }
        PICKED_FRAME[3].fetch_add(1, Ordering::Relaxed);
        if !reads.is_empty() {
            PICKED_FRAME[4].fetch_add(1, Ordering::Relaxed);
        }
        (reads, writes)
    }

    /// The pipeline a shader writing through picked slots belongs to.
    pub(crate) fn writer(&self) -> u64 {
        self.writer.load(Ordering::Relaxed)
    }

    /// What `submission`'s GPU work touches of the heap's listing, once
    /// per submission ([`Listing::touches`]): title ranges it reads, and
    /// runs its shaders write.
    pub(crate) fn touches(&self, submission: u64) -> (Vec<(usize, usize)>, Vec<Span>) {
        self.picked.listing().touches(submission)
    }
}

/// What a recording of a list did with a heap's listing
/// ([`HostHeap::picked_spans`]): its step for the list's current refresh
/// epoch ([`crate::host::HostList::refresh_epoch`]). A draw or dispatch that
/// picks slots adds nothing to the list but that step, once per epoch: the
/// runs are given at submission ([`PickedStep`]).
///
/// Codex review of 8612e27 (stride round 4b), P2: each recording caught up
/// on the listing from nothing, its first pick adding a refresh step per run
/// of the heap (5,435 runs, some 42 lists a frame: 226,575 refresh steps),
/// each settled and merged at submission. And the steps the sweep at each
/// submission added stayed in the list (a closed list resubmitted with
/// another buffer in a slot each time refreshed every buffer it had seen,
/// and held it, until its reset), and a copy's runs were copied whole at
/// each pick (N disjoint views of one copy, N(N+1)/2 runs looked at). Now
/// what a submission is given is worked out at submission, held by the
/// heap's listing, not the list.
#[derive(Default)]
pub(crate) struct PickedUse {
    /// A shader of the recording writes through slots it picks.
    pub(crate) writing: bool,
    /// The recording has another heap under the key now (`draw.rs`
    /// `forget_heap`): nothing more is done for this one.
    pub(crate) forgotten: bool,
    /// The step for refresh epoch `.0`.
    step: Option<(u64, Arc<PickedStep>)>,
}

/// Since the start, for the sweep's log: runs submissions were given to
/// refresh ([`PickedStep::due`]), their bytes, and runs given to declare
/// written.
static PICKS: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];

/// Since the last frame line ([`log_picked_frame`]): as [`PICKS`], then
/// the steps run and those given runs.
static PICKED_FRAME: [AtomicU64; 5] = [const { AtomicU64::new(0) }; 5];

/// The most raw views of a stride table's pool taken at once, since the
/// start (any heap), and the raw views refused.
static POOL_PEAK: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
static POOL_REFUSED: AtomicU64 = AtomicU64::new(0);

/// The frame line's account of what shaders that pick slots at run time
/// cost the submissions (`crate::host`'s frame log): steps, runs and bytes
/// given to refresh, runs declared written, per frame; and the raw views'
/// pool high-water mark. Nothing without such steps.
pub(crate) fn log_picked_frame(frames: f64) {
    let [runs, bytes, declared, steps, giving] = PICKED_FRAME
        .each_ref()
        .map(|n| n.swap(0, Ordering::Relaxed));
    if steps == 0 {
        return;
    }
    let per = |n: u64| n as f64 / frames;
    tracing::info!(
        "slots picked at run time per frame: {:.1} steps ({:.1} given runs), {:.1} runs refreshed \
         ({:.1} MB), {:.1} runs declared written; raw view pool peak {} ({} refused so far)",
        per(steps),
        per(giving),
        per(runs),
        per(bytes) / 1e6,
        per(declared),
        POOL_PEAK.load(Ordering::Relaxed),
        POOL_REFUSED.load(Ordering::Relaxed),
    );
}

impl PickedUse {
    /// A draw or dispatch of `list` binds a shader that picks slots of the
    /// heap listed in `p` (and, `writes`, writes through them): the list's
    /// step for its refresh epoch, made at the first such draw of the epoch
    /// (a title copy or work of the list starts another, after which every
    /// run is brought up to date again), marked writing.
    pub(crate) fn defer(&mut self, list: &crate::host::HostList, p: &Arc<Picked>, writes: bool) {
        self.writing |= writes;
        let epoch = list.refresh_epoch();
        let step = match &self.step {
            Some((at, step)) if *at == epoch && Arc::ptr_eq(&step.picked, p) => step.clone(),
            _ => {
                let step = Arc::new(PickedStep {
                    picked: p.clone(),
                    writes: std::sync::atomic::AtomicBool::new(false),
                    writer: AtomicU64::new(0),
                });
                list.defer_picked(step.clone());
                self.step = Some((epoch, step.clone()));
                step
            }
        };
        if writes && !step.writes.load(Ordering::Relaxed) {
            step.writer.store(list.writer(), Ordering::Relaxed);
            step.writes.store(true, Ordering::Relaxed);
        }
    }
}

/// Time a frame's [`HostHeap::picked_spans`] sweeps may spend making host
/// copies before it is logged: making one is a resource, which can wait in
/// the kernel, and the world's loading leaves tens of thousands of views
/// waiting. They are all made all the same (Codex review of 244cf78,
/// finding 6: past this time the views left stayed null, and a dispatch
/// that picked one read 0).
const PICKED_BUDGET: std::time::Duration = std::time::Duration::from_millis(8);

/// The sweeps of a heap's buffer views for shaders that pick slots at run
/// time ([`HostHeap::picked_spans`]).
#[derive(Default)]
struct Sweeps {
    /// The frame of the last sweep, when its time began, and the time its
    /// sweeps spent making host copies.
    frame: u64,
    began: Option<std::time::Instant>,
    spent: std::time::Duration,
}

/// Whether buffer views of title slots are kept for shaders that pick
/// slots when they run (`XTP_RUN_TIME_VIEWS=1`,
/// [`crate::device::run_time_views_on`]): the views the title writes wait
/// for a sweep, and those made are listed by host copy. Off, nothing is
/// kept, as before the switch. Tests set it per thread
/// ([`RUN_TIME_VIEWS_TEST`]).
fn run_time_views() -> bool {
    #[cfg(test)]
    if let Some(on) = RUN_TIME_VIEWS_TEST.get() {
        return on;
    }
    crate::device::run_time_views_on()
}

#[cfg(test)]
thread_local! {
    /// [`run_time_views`] for this thread's tests.
    pub(crate) static RUN_TIME_VIEWS_TEST: std::cell::Cell<Option<bool>> =
        const { std::cell::Cell::new(None) };
}

/// Dynamic slot bookkeeping.
struct Slots {
    /// Maximal free runs, indexed by start for merging and by length for best fit.
    by_start: BTreeMap<u32, u32>,
    by_size: BTreeSet<(u32, u32)>,
    /// Slots taken now, and the most ever taken at once.
    used: u32,
    peak: u32,
}

/// Terminal allocation failures since the last frame boundary.
#[cfg(feature = "diagnostics")]
static FRAME_FAILURES: AtomicU64 = AtomicU64::new(0);

#[cfg(feature = "diagnostics")]
pub(crate) fn log_frame_failures(frame: u64) {
    let failures = FRAME_FAILURES.swap(0, Ordering::Relaxed);
    if failures != 0 || frame.is_multiple_of(60) {
        tracing::info!(
            frame,
            failures,
            "dynamic descriptor allocation failures this frame"
        );
    }
}

impl Slots {
    fn new(capacity: u32) -> Self {
        let mut slots = Self {
            by_start: BTreeMap::new(),
            by_size: BTreeSet::new(),
            used: 0,
            peak: 0,
        };
        slots.insert_run(0, capacity);
        slots
    }

    fn insert_run(&mut self, first: u32, length: u32) {
        if length > 0 {
            self.by_start.insert(first, length);
            self.by_size.insert((length, first));
        }
    }

    fn remove_run(&mut self, first: u32, length: u32) {
        self.by_start.remove(&first);
        self.by_size.remove(&(length, first));
    }

    /// Use the smallest sufficient hole, leaving larger tables room.
    fn reserve(&mut self, n: u32) -> Option<u32> {
        let &(length, first) = self.by_size.range((n, 0)..).next()?;
        self.remove_run(first, length);
        self.insert_run(first + n, length - n);
        self.used += n;
        Some(first)
    }

    /// Merge touching or overlapping free runs. Duplicate or partial releases
    /// count only the slots that were still held.
    fn release(&mut self, mut first: u32, mut end: u32) {
        if first >= end {
            return;
        }
        if let Some((&left, &len)) = self.by_start.range(..first).next_back()
            && left + len >= first
        {
            first = left;
        }
        let mut already_free = 0;
        while let Some((&next, &len)) = self.by_start.range(first..=end).next() {
            self.remove_run(next, len);
            already_free += len;
            end = end.max(next + len);
        }
        self.used -= end - first - already_free;
        self.insert_run(first, end - first);
    }

    fn failure(&self, requested: u32, capacity: u32) {
        #[cfg(feature = "diagnostics")]
        FRAME_FAILURES.fetch_add(1, Ordering::Relaxed);
        static FAILED: AtomicU64 = AtomicU64::new(0);
        let failed = FAILED.fetch_add(1, Ordering::Relaxed) + 1;
        if failed.is_power_of_two() {
            let largest_free_run = self.by_size.last().map_or(0, |&(len, _)| len);
            tracing::warn!(
                requested,
                free = capacity - self.used,
                largest_free_run,
                used = self.used,
                peak = self.peak,
                of = capacity,
                failed,
                "no free dynamic descriptor slots: a table or view is not made"
            );
        }
    }
}

/// Slots per chunk the shared views take at once.
pub(crate) const SHARED_CHUNK: u32 = 256;

/// Shared views no list holds that are kept for lists to come; past it the
/// least recently held half goes.
const IDLE_SHARED: usize = 4096;

/// A range of a host copy: the copy, the byte offset, the length.
type SharedKey = (usize, u64, u64);

/// Shared raw views of host copies (see [`HostHeap::shared_raw`]).
#[derive(Default)]
struct Shared {
    index: HashMap<SharedKey, u32>,
    entries: HashMap<u32, SharedView>,
    /// Chunks of dynamic slots: first slot, the free slots in it.
    chunks: Vec<(u32, Vec<u32>)>,
    /// Entries no list holds.
    idle: usize,
    /// Counts lists giving views back, to tell how long ago an entry was
    /// last held.
    tick: u64,
}

struct SharedView {
    key: SharedKey,
    /// The lists holding it (not yet reset).
    holders: u32,
    /// When it was last given back.
    last: u64,
    /// The host copy it views: an entry whose copy is gone (the title
    /// released the buffer) goes first.
    copy: Weak<Shadow>,
}

// SAFETY: D3D12 descriptor heaps and resources are free-threaded; the rest
// of the state is behind mutexes. Needed for XTP's own view heap, a static.
unsafe impl Send for HostHeap {}
// SAFETY: as above.
unsafe impl Sync for HostHeap {}

/// Host heaps' GPU ranges (start, bytes, bytes per slot, sampler), for
/// device-removed reports: a shader indexing a heap past its end faults.
static GPU_RANGES: Mutex<Vec<(u64, u64, u64, bool)>> = Mutex::new(Vec::new());

/// For device-removed reports: where host GPU address `va` lies against
/// the host descriptor heaps.
pub(crate) fn near_heaps(va: u64) -> String {
    let ranges = GPU_RANGES.lock().unwrap_or_else(|e| e.into_inner());
    let mut text = String::new();
    for &(start, bytes, step, sampler) in ranges.iter() {
        let kind = if sampler { "sampler" } else { "view" };
        let place = if va < start {
            format!("{:#x} below", start - va)
        } else if va < start + bytes {
            format!("inside, slot {}", (va - start) / step)
        } else {
            format!(
                "{:#x} past its end (slot {})",
                va - start - bytes,
                (va - start) / step
            )
        };
        text.push_str(&format!("; {kind} heap {start:#x}+{bytes:#x}: {place}"));
    }
    text
}

/// The host's limit for shader-visible view heaps on resource binding
/// tiers 1 and 2; tier 3 hardware may take more. An RTX 4080 SUPER (tier
/// 3) takes exactly this many, not one more (2026-11-24).
const PORTABLE_VIEWS: u32 = 1_000_000;

/// Slots past the title's in a view heap, for tables a draw needs made
/// afresh (constant buffers copied per draw). A list holds the slots it
/// took until it is reset (its submissions have run) or released.
/// 2^16 ran out seconds into the world (every world run on an RX 9070 XT,
/// each a device removal shortly after; #59), and on capped heaps too:
/// NVIDIA worlds stalled (VIctory #747), and XTP's Series S world on the
/// RTX 4080 reached 57,364 of 65,536 and dropped draws for want of slots.
const DYNAMIC: u32 = 1 << 18;

/// The dynamic slots of a heap that the host would not make with
/// [`DYNAMIC`] more but would with these, which leave every title slot
/// in.
const MIN_DYNAMIC: u32 = 1 << 16;

/// A capped view heap of `limit` host slots split into (title slots
/// mirrored, dynamic slots): [`DYNAMIC`] dynamic ones, at least one title
/// slot when `limit` > 1. The title's slots past the first part read
/// nothing; BTP 6's view heap has 1,245,184, and none of its runs wrote a
/// slot past 934,464 (the part 2^16 dynamic slots leave).
fn capped_split(limit: u32) -> (u32, u32) {
    let dynamic = DYNAMIC.min(limit.saturating_sub(1));
    (limit - dynamic, dynamic)
}

/// D3D12 limits a raw buffer view to 2^27 four-byte elements.
pub(crate) const MAX_RAW_VIEW_BYTES: u64 = 4 << D3D12_REQ_BUFFER_RESOURCE_TEXEL_COUNT_2_TO_EXP;

impl HostHeap {
    /// A host heap for a title heap of `count` slots; view heaps get a
    /// runtime stride table with `XTP_STRIDE_TABLE=1`.
    pub(crate) fn new(sampler: bool, count: u32) -> Option<Self> {
        Self::new_with(sampler, count, crate::device::stride_table_on())
    }

    /// [`HostHeap::new`], with a runtime stride table (view heaps only) or
    /// not: [`ALIASES`] more slots for its raw views, on top of the dynamic
    /// ones, and counted in the host's cap.
    pub(crate) fn new_with(sampler: bool, count: u32, stride_table: bool) -> Option<Self> {
        let aliases = if sampler || !stride_table { 0 } else { ALIASES };
        Self::new_pooled(sampler, count, aliases)
    }

    /// A view heap with a stride table whose pool has `aliases` raw views
    /// (tests that fill it).
    #[cfg(test)]
    fn with_pool(count: u32, aliases: u32) -> Option<Self> {
        Self::new_pooled(false, count, aliases)
    }

    /// [`HostHeap::new_with`] with `aliases` slots for the stride table's
    /// raw views (0: no table).
    fn new_pooled(sampler: bool, count: u32, aliases: u32) -> Option<Self> {
        let host = host()?;
        let kind = if sampler {
            D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER
        } else {
            D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV
        };
        let make = |count: u32| -> windows::core::Result<ID3D12DescriptorHeap> {
            // SAFETY: plain heap creation.
            unsafe {
                host.device
                    .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                        Type: kind,
                        NumDescriptors: count,
                        Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                        NodeMask: 0,
                    })
            }
        };
        // The larger dynamic range where the host takes it on top of every
        // title slot; else the smaller one, which keeps every title slot;
        // past the host's cap, the larger one again (VIctory #1349).
        let (heap, count, extra) = if sampler {
            match make(count) {
                Ok(heap) => (heap, count, 0),
                Err(e) => {
                    tracing::warn!(error = %e, count, sampler, "no host descriptor heap");
                    return None;
                }
            }
        } else if let Ok(heap) = make(count + DYNAMIC + aliases) {
            (heap, count, DYNAMIC)
        } else {
            match make(count + MIN_DYNAMIC + aliases) {
                Ok(heap) => (heap, count, MIN_DYNAMIC),
                Err(e) if count + MIN_DYNAMIC + aliases > PORTABLE_VIEWS => {
                    // shortcut: descriptors past the host's limit are left
                    // out; a title that reaches them needs remapping ("title
                    // slot past the host heap" says so). The raw views of
                    // the stride table come out of the dynamic part (2^18
                    // less 2^16 left; the Series S world's peak in use was
                    // 98,312), not the title's.
                    let (mirrored, dynamic) = capped_split(PORTABLE_VIEWS);
                    let dynamic = dynamic.saturating_sub(aliases);
                    tracing::warn!(
                        error = %e,
                        count,
                        mirrored,
                        dynamic,
                        aliases,
                        "host view heap capped at {PORTABLE_VIEWS}"
                    );
                    (make(PORTABLE_VIEWS).ok()?, mirrored, dynamic)
                }
                Err(e) => {
                    tracing::warn!(error = %e, count, sampler, "no host descriptor heap");
                    return None;
                }
            }
        };
        Self::from_heap(heap, kind, sampler, count, extra, aliases)
    }

    /// XTP's own view heap of `dynamic` dynamic slots and no title slots,
    /// bound in the view slot of lists that bind no hosted view heap of the
    /// title's (root SRVs are read through tables in it).
    pub(crate) fn own(dynamic: u32) -> Option<Self> {
        let host = host()?;
        let kind = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        // SAFETY: plain heap creation.
        let made = unsafe {
            host.device
                .CreateDescriptorHeap::<ID3D12DescriptorHeap>(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: kind,
                    NumDescriptors: dynamic,
                    Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                    NodeMask: 0,
                })
        };
        match made {
            Ok(heap) => Self::from_heap(heap, kind, false, 0, dynamic, 0),
            Err(e) => {
                tracing::warn!(error = %e, dynamic, "no view heap of XTP's own");
                None
            }
        }
    }

    /// XTP's own sampler heap: a few null samplers, bound so that a root
    /// signature indexing the sampler heap directly may be set in a list
    /// that binds no sampler heap of the title's.
    pub(crate) fn own_samplers() -> Option<Self> {
        const SLOTS: u32 = 16;
        let host = host()?;
        let kind = D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER;
        // SAFETY: plain heap creation.
        let made = unsafe {
            host.device
                .CreateDescriptorHeap::<ID3D12DescriptorHeap>(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: kind,
                    NumDescriptors: SLOTS,
                    Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                    NodeMask: 0,
                })
        };
        match made {
            Ok(heap) => Self::from_heap(heap, kind, true, 0, SLOTS, 0),
            Err(e) => {
                tracing::warn!(error = %e, "no sampler heap of XTP's own");
                None
            }
        }
    }

    /// A host heap around `heap`: `count` title slots, then `extra`
    /// dynamic ones, then `aliases` for the raw views of a runtime stride
    /// table (none: no table), every slot made null.
    fn from_heap(
        heap: ID3D12DescriptorHeap,
        kind: D3D12_DESCRIPTOR_HEAP_TYPE,
        sampler: bool,
        count: u32,
        extra: u32,
        aliases: u32,
    ) -> Option<Self> {
        let host = host()?;
        // SAFETY: a live heap.
        let (cpu, step) = unsafe {
            (
                heap.GetCPUDescriptorHandleForHeapStart().ptr,
                host.device.GetDescriptorHandleIncrementSize(kind),
            )
        };
        tracing::debug!(count, extra, aliases, sampler, "host descriptor heap");
        // SAFETY: a live shader-visible heap.
        let gpu = unsafe { heap.GetGPUDescriptorHandleForHeapStart().ptr };
        let total = count + extra + aliases;
        GPU_RANGES.lock().unwrap_or_else(|e| e.into_inner()).push((
            gpu,
            u64::from(total) * u64::from(step),
            u64::from(step),
            sampler,
        ));
        // Every slot of the heap has an entry: a shader reads the one of
        // whatever slot it computes, if it is inside the table.
        let strides = (aliases > 0)
            .then(|| StrideTable::new(&host.device, total, count))
            .flatten()
            .map(|table| {
                Arc::new(Strides {
                    table,
                    alias_base: count + extra,
                    mirrored: count,
                    _heap: heap.clone(),
                    cpu,
                    step,
                    aliases: Mutex::new(Aliases {
                        size: aliases,
                        ..Aliases::default()
                    }),
                })
            });
        let listing = Arc::new(Mutex::new(Listing::default()));
        let h = HostHeap {
            heap,
            cpu,
            step,
            count,
            dynamic: extra,
            sampler,
            made: Mutex::new(Made {
                keys: vec![0; count as usize],
                textures: HashMap::new(),
                waiting: Vec::new(),
                queued: vec![false; if sampler { 0 } else { count as usize }],
                buffers: HashMap::new(),
                listing: listing.clone(),
            }),
            kept: Mutex::new(vec![None; (count + extra) as usize]),
            slots: Mutex::new(Slots::new(extra)),
            shared: Mutex::default(),
            null_raw: Mutex::new(None),
            strides,
            picked: Mutex::default(),
            listed: Arc::new(Picked { listing }),
            textures: Mutex::default(),
            tabled: (0..if sampler { 0 } else { count })
                .map(|_| std::sync::atomic::AtomicU32::new(0))
                .collect(),
            passes_on: AtomicU64::new(0),
        };
        // A slot never written is undefined: shaders reading one (a slot
        // the title wrote in a way we do not know, or a table entry no
        // used-resource entry names) read a wild address. Every slot starts
        // null instead.
        let started = std::time::Instant::now();
        if let Err(e) = h.fill_null(kind, total) {
            tracing::warn!(error = %e, "host descriptor heap not made null");
        }
        tracing::debug!(
            ms = started.elapsed().as_millis(),
            "host descriptor heap made null"
        );
        Some(h)
    }

    /// Makes the first `total` slots null: one null slot in a CPU-side heap,
    /// doubled by copies to a block, the block copied over the heap.
    fn fill_null(&self, kind: D3D12_DESCRIPTOR_HEAP_TYPE, total: u32) -> windows::core::Result<()> {
        const BLOCK: u32 = 1 << 16;
        let Some(host) = host() else { return Ok(()) };
        // SAFETY: plain heap creation and copies inside live heaps.
        unsafe {
            let block: ID3D12DescriptorHeap =
                host.device
                    .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                        Type: kind,
                        NumDescriptors: BLOCK,
                        ..Default::default()
                    })?;
            let start = block.GetCPUDescriptorHandleForHeapStart();
            let at = |i: u32| D3D12_CPU_DESCRIPTOR_HANDLE {
                ptr: start.ptr + (i * self.step) as usize,
            };
            if self.sampler {
                null_sampler(start);
            } else {
                null_view(start);
            }
            let mut filled = 1;
            while filled < BLOCK {
                let n = filled.min(BLOCK - filled);
                host.device
                    .CopyDescriptorsSimple(n, at(filled), start, kind);
                filled += n;
            }
            let mut done = 0;
            while done < total {
                let n = BLOCK.min(total - done);
                host.device
                    .CopyDescriptorsSimple(n, self.cpu(done), start, kind);
                done += n;
            }
        }
        Ok(())
    }

    /// A plain point sampler at slot `index`, for sampler slots without a
    /// view.
    fn null_sampler(&self, index: u32) {
        null_sampler(self.cpu(index));
    }
}

/// A plain point sampler at `at`.
fn null_sampler(at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    let Some(host) = host() else { return };
    let desc = D3D12_SAMPLER_DESC {
        Filter: D3D12_FILTER_MIN_MAG_MIP_POINT,
        AddressU: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressV: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressW: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        MaxAnisotropy: 1,
        ComparisonFunc: D3D12_COMPARISON_FUNC_NEVER,
        ..Default::default()
    };
    // SAFETY: a slot of a sampler heap of ours.
    unsafe { host.device.CreateSampler(&desc, at) };
}

impl HostHeap {
    /// `n` consecutive free dynamic slots (view heaps only), taken until
    /// [`HostHeap::release`]: the first's index. When none are free, the
    /// shared views no list holds give theirs back first.
    pub(crate) fn dynamic(&self, n: u32) -> Option<u32> {
        if n > self.dynamic && !self.sampler {
            return self.try_take(n, true);
        }
        if let Some(first) = self.take(n) {
            return Some(first);
        }
        if self.sampler || n == 0 {
            return None;
        }
        {
            let mut shared = self.shared.lock().unwrap_or_else(|e| e.into_inner());
            let idle = shared.idle;
            if idle > 0 {
                self.evict(&mut shared, idle);
            }
        }
        self.try_take(n, true)
    }

    /// An allocation attempt before reclaiming idle shared views.
    fn take(&self, n: u32) -> Option<u32> {
        self.try_take(n, false)
    }

    fn try_take(&self, n: u32, final_attempt: bool) -> Option<u32> {
        if self.sampler || n == 0 {
            return None;
        }
        let mut guard = self.slots.lock().unwrap_or_else(|e| e.into_inner());
        let s = &mut *guard;
        let Some(first) = s.reserve(n) else {
            if final_attempt {
                s.failure(n, self.dynamic);
            }
            return None;
        };
        if s.used > s.peak {
            // Logged at each new eighth of the range.
            let eighth = |v: u32| u64::from(v) * 8 / u64::from(self.dynamic);
            if eighth(s.used) > eighth(s.peak) {
                tracing::info!(
                    peak = s.used,
                    of = self.dynamic,
                    "dynamic descriptor slots: a new peak in use"
                );
            }
            s.peak = s.used;
        }
        Some(self.count + first)
    }

    /// The dynamic slots taken now, and the most ever taken at once.
    #[cfg(test)]
    pub(crate) fn dynamic_use(&self) -> (u32, u32) {
        let s = self.slots.lock().unwrap_or_else(|e| e.into_inner());
        (s.used, s.peak)
    }

    /// The title slots this heap has host slots for; fewer than the title's
    /// when the host caps its size.
    pub(crate) fn count(&self) -> u32 {
        self.count
    }

    /// Gives back `n` dynamic slots from `first` (an index [`dynamic`]
    /// returned).
    ///
    /// [`dynamic`]: HostHeap::dynamic
    #[cfg(test)]
    pub(crate) fn release(&self, first: u32, n: u32) {
        self.release_all(&[(first, n)]);
    }

    /// Gives back each range `(first, n)` of dynamic slots (indices
    /// [`HostHeap::dynamic`] returned), under one lock.
    pub(crate) fn release_all(&self, ranges: &[(u32, u32)]) {
        if ranges.is_empty() {
            return;
        }
        // Their stride table entries and raw views go first: once free, a
        // slot may be taken by another list, which writes its own (VIctory
        // #704, 4e03086).
        self.forget_ranges(ranges);
        let mut guard = self.slots.lock().unwrap_or_else(|e| e.into_inner());
        let s = &mut *guard;
        for &(first, n) in ranges {
            let Some(at) = first.checked_sub(self.count) else {
                continue;
            };
            s.release(at, at.saturating_add(n).min(self.dynamic));
        }
    }

    /// A raw view of `len` bytes of `copy` from byte `offset` (a multiple
    /// of 16; see [`HostHeap::raw_at`]): its slot, shared by every list
    /// that asks for the same range and held once more for the caller
    /// until [`HostHeap::give_back`]. Shared views are made once, not once
    /// per list, which spares the dynamic slots: a capped heap (NVIDIA) has
    /// 2^16 of them for every list's fresh tables too. `None` if no slot is
    /// free.
    pub(crate) fn shared_raw(&self, copy: &Arc<Shadow>, offset: u64, len: u64) -> Option<u32> {
        let key = (copy.buffer.as_raw() as usize, offset, len);
        let mut guard = self.shared.lock().unwrap_or_else(|e| e.into_inner());
        let s = &mut *guard;
        if let Some(&slot) = s.index.get(&key)
            && let Some(entry) = s.entries.get_mut(&slot)
        {
            if entry.holders == 0 {
                s.idle -= 1;
            }
            entry.holders += 1;
            return Some(slot);
        }
        let slot = match self.shared_slot(s, false) {
            Some(slot) => slot,
            None => {
                let idle = s.idle;
                self.evict(s, idle);
                self.shared_slot(s, true)?
            }
        };
        self.raw_at(slot, &copy.buffer, offset, len);
        s.index.insert(key, slot);
        s.entries.insert(
            slot,
            SharedView {
                key,
                holders: 1,
                last: s.tick,
                copy: Arc::downgrade(copy),
            },
        );
        Some(slot)
    }

    /// A free slot of the shared views' chunks, taking a new chunk if
    /// they have none.
    fn shared_slot(&self, s: &mut Shared, final_attempt: bool) -> Option<u32> {
        if let Some(slot) = s.chunks.iter_mut().find_map(|(_, free)| free.pop()) {
            return Some(slot);
        }
        let first = self.try_take(SHARED_CHUNK, final_attempt)?;
        // Handed out from the chunk's start.
        s.chunks
            .push((first, (first + 1..first + SHARED_CHUNK).rev().collect()));
        Some(first)
    }

    /// The lists giving back shared views they held (slots
    /// [`HostHeap::shared_raw`] returned), under one lock. Views no list
    /// holds stay for later lists, up to [`IDLE_SHARED`] of them.
    pub(crate) fn give_back(&self, slots: &[u32]) {
        if slots.is_empty() {
            return;
        }
        let mut guard = self.shared.lock().unwrap_or_else(|e| e.into_inner());
        let s = &mut *guard;
        s.tick += 1;
        for slot in slots {
            if let Some(entry) = s.entries.get_mut(slot)
                && entry.holders > 0
            {
                entry.holders -= 1;
                if entry.holders == 0 {
                    entry.last = s.tick;
                    s.idle += 1;
                }
            }
        }
        if s.idle > IDLE_SHARED {
            let n = s.idle - IDLE_SHARED / 2;
            self.evict(s, n);
        }
    }

    /// Drops `n` shared views no list holds: those whose host copy is gone
    /// first, then the least recently held. Chunks left empty go back to
    /// the dynamic slots.
    fn evict(&self, s: &mut Shared, n: usize) {
        let mut idle: Vec<(bool, u64, u32)> = s
            .entries
            .iter()
            .filter(|(_, e)| e.holders == 0)
            .map(|(&slot, e)| (e.copy.strong_count() > 0, e.last, slot))
            .collect();
        idle.sort_unstable();
        for &(_, _, slot) in idle.iter().take(n) {
            let Some(entry) = s.entries.remove(&slot) else {
                continue;
            };
            s.index.remove(&entry.key);
            s.idle -= 1;
            // Nothing reads it (every list that held it was reset): made
            // null, it no longer holds the host copy.
            self.null_raw_at(slot);
            if let Some((_, free)) = s
                .chunks
                .iter_mut()
                .find(|(first, _)| (*first..*first + SHARED_CHUNK).contains(&slot))
            {
                free.push(slot);
            }
        }
        let mut empty = Vec::new();
        s.chunks.retain(|(first, free)| {
            let whole = free.len() == SHARED_CHUNK as usize;
            if whole {
                empty.push((*first, SHARED_CHUNK));
            }
            !whole
        });
        self.release_all(&empty);
    }

    /// The shared views (all, and those no list holds).
    #[cfg(test)]
    pub(crate) fn shared_count(&self) -> (usize, usize) {
        let s = self.shared.lock().unwrap_or_else(|e| e.into_inner());
        (s.entries.len(), s.idle)
    }

    /// A slot holding a raw view that reads zeros, the same for every list
    /// (never given back); `None` if no slot is free.
    pub(crate) fn null_raw(&self) -> Option<u32> {
        let mut cached = self.null_raw.lock().unwrap_or_else(|e| e.into_inner());
        if cached.is_none() {
            let slot = self.dynamic(1)?;
            self.null_raw_at(slot);
            *cached = Some(slot);
        }
        *cached
    }

    /// Makes slot `index` (a dynamic one) from `view`; a null view if the
    /// host cannot. True if the host made the view.
    pub(crate) fn make_at(&self, index: u32, view: &View) -> bool {
        let mut viewed = None;
        let wanted = self.strides.is_some().then_some(&mut viewed);
        let kept = make_with(view, self.cpu(index), wanted);
        let made = kept.is_some();
        if !made {
            null_for(view, self.cpu(index));
        }
        // (A view the host could not make may still give the table its
        // bytes: `off_element_view`. A view the full pool gives no raw view
        // gets one as soon as one comes back, `Aliases::refused_slots`: the
        // list may keep this table for later draws.)
        self.note_view(index, viewed.as_ref());
        self.keep(index, kept.flatten());
        made
    }

    /// Makes slot `index` read nothing: a null view, or a plain sampler.
    pub(crate) fn make_null(&self, index: u32) {
        if self.sampler {
            self.null_sampler(index);
        } else {
            null_view(self.cpu(index));
        }
        self.forget_view(index);
        self.keep(index, None);
    }

    /// Slot `index` now views `resource` (or nothing of the host's).
    fn keep(&self, index: u32, resource: Option<ID3D12Resource>) {
        // The one it held goes after the lock: its last reference may free
        // video memory, which can wait.
        drop(self.swap_kept(index, resource));
    }

    /// The actual resource of a host descriptor, including a freshly
    /// remapped or placed view. Null descriptors have no resource.
    pub(crate) fn kept_resource(&self, index: u32) -> Option<ID3D12Resource> {
        self.kept
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(index as usize)
            .cloned()
            .flatten()
    }

    /// [`HostHeap::keep`], handing back the resource the slot held.
    fn swap_kept(&self, index: u32, resource: Option<ID3D12Resource>) -> Option<ID3D12Resource> {
        let mut kept = self.kept.lock().unwrap_or_else(|e| e.into_inner());
        kept.get_mut(index as usize)
            .and_then(|slot| std::mem::replace(slot, resource))
    }

    /// Makes slot `index` a null constant buffer view: it reads zeros, as
    /// the console's empty or zero-sized constant views do. (A null 2D
    /// texture view where the shader declares a constant buffer is a type
    /// mismatch GPU-based validation reports.)
    pub(crate) fn null_constant(&self, index: u32) {
        self.forget_view(index);
        self.keep(index, None);
        let Some(host) = host() else { return };
        // SAFETY: a slot of our heap; no description makes a null view.
        unsafe { host.device.CreateConstantBufferView(None, self.cpu(index)) };
    }

    /// Makes slot `index` (a dynamic one) a constant buffer view of host
    /// memory (the list's own, which it holds): [`constant_bytes`] of it,
    /// whose bytes past `size` the caller zeroed. A view of nothing (no
    /// address or no bytes) is a null one.
    pub(crate) fn constant_at(&self, index: u32, gpu: u64, size: u32) {
        if gpu == 0 || size == 0 {
            self.null_constant(index);
            return;
        }
        self.forget_view(index);
        self.keep(index, None);
        let Some(host) = host() else { return };
        let desc = D3D12_CONSTANT_BUFFER_VIEW_DESC {
            BufferLocation: gpu,
            SizeInBytes: constant_bytes(size),
        };
        // SAFETY: a slot of our heap; the memory is the caller's.
        unsafe {
            host.device
                .CreateConstantBufferView(Some(&desc), self.cpu(index))
        };
    }

    /// Makes slot `index` (a dynamic one) a raw buffer view of `len` bytes
    /// of `buffer` from byte `offset` (a multiple of 16): a root SRV with a
    /// size, whose reads past it return zeros instead of reaching whatever
    /// the host placed after the buffer.
    ///
    /// The last dword is whole even when `len` is not a multiple of 4 (host
    /// copies are rounded up to 64 KiB, so it is inside `buffer`), and the
    /// view stops at D3D12's 2^27 buffer view elements (512 MiB).
    pub(crate) fn raw_at(&self, index: u32, buffer: &ID3D12Resource, offset: u64, len: u64) {
        let Some(host) = host() else { return };
        let elements = len.div_ceil(4).clamp(1, MAX_RAW_VIEW_BYTES / 4) as u32;
        let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT_R32_TYPELESS,
            ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Buffer: D3D12_BUFFER_SRV {
                    FirstElement: offset / 4,
                    NumElements: elements,
                    StructureByteStride: 0,
                    Flags: D3D12_BUFFER_SRV_FLAG_RAW,
                },
            },
        };
        // SAFETY: a slot of our heap and a live buffer covering the range.
        unsafe {
            host.device
                .CreateShaderResourceView(buffer, Some(&desc), self.cpu(index))
        };
        // A root SRV's view: no table entry names it (no title slot does).
        self.forget_view(index);
        self.keep(index, Some(buffer.clone()));
    }

    pub(crate) fn raw_uav_at(&self, index: u32, buffer: &ID3D12Resource, window: RawWindow) {
        raw_view(buffer, window, true, self.cpu(index));
        self.forget_view(index);
        self.keep(index, Some(buffer.clone()));
    }

    /// Makes slot `index` a raw buffer view that reads zeros.
    pub(crate) fn null_raw_at(&self, index: u32) {
        null_buffer_view(self.cpu(index));
        self.forget_view(index);
        self.keep(index, None);
    }

    /// The shader-visible handle of slot `index`.
    pub(crate) fn gpu_handle(&self, index: u32) -> D3D12_GPU_DESCRIPTOR_HANDLE {
        // SAFETY: a live shader-visible heap.
        let start = unsafe { self.heap.GetGPUDescriptorHandleForHeapStart().ptr };
        D3D12_GPU_DESCRIPTOR_HANDLE {
            ptr: start + u64::from(index) * u64::from(self.step),
        }
    }

    fn cpu(&self, index: u32) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        D3D12_CPU_DESCRIPTOR_HANDLE {
            ptr: self.cpu + index as usize * self.step as usize,
        }
    }

    /// Makes host slot `index` from the record the title has at `handle`
    /// if it changed since, and returns the view recorded there. Buffer
    /// views are made only for draws (`for_draw`), which also copy their
    /// data: a host copy of a buffer is made on first use.
    /// shortcut: the slot is written at once, though lists submitted before
    /// may still read it on the GPU (D3D12 asks that a descriptor in use
    /// stay as it is). A title that rewrites a slot does as much on the
    /// console; a slot made again because its host texture changed (made
    /// resident, replaced: #643) is not the title's doing, and such lists
    /// may read the view before or after (both of the same kind: a null
    /// view is made in the title's dimension).
    ///
    /// The record's three words are read whole, under the lock its writers
    /// take ([`super::header`]), while another thread may be publishing one
    /// there (the title's view creation and descriptor copies, which call
    /// this after their write): the slot is made only if they still hold
    /// what was read once the slot's own lock is taken, else read again
    /// ([`MIRROR_TRIES`] times, then left to the writer's own call). A
    /// record read before the write is never made after it.
    pub(crate) fn mirror(&self, index: u32, handle: u64, for_draw: bool) -> Option<View> {
        for _ in 0..MIRROR_TRIES {
            match self.mirror_once(index, handle, for_draw, true) {
                Ok(view) => return view,
                Err(Unmade::Rewritten) => continue,
                Err(Unmade::NoCopy) => return None,
            }
        }
        None
    }

    /// One try of [`HostHeap::mirror`]; with `copies` false, a buffer view
    /// whose host copy is not made yet is left as it is (`NoCopy`: the
    /// sweep makes it after the views whose copy is there,
    /// [`HostHeap::picked_spans`]).
    fn mirror_once(
        &self,
        index: u32,
        handle: u64,
        for_draw: bool,
        copies: bool,
    ) -> Result<Option<View>, Unmade> {
        if index >= self.count {
            // Diagnostic (NVIDIA heap cap): title slots past the host heap,
            // which are not mirrored, counted apart for the title's writes
            // and for draws reading one through a table (it reads nothing);
            // logged at each new highest slot and at each power of two.
            // shortcut: shaders reading such a slot by heap index (not seen
            // here) read XTP's dynamic slots, or past the heap.
            static HIGHEST: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            static PAST: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];
            let higher = HIGHEST.fetch_max(index, Ordering::Relaxed) < index;
            let n = PAST[usize::from(for_draw)].fetch_add(1, Ordering::Relaxed) + 1;
            if higher || n.is_power_of_two() {
                tracing::warn!(
                    index,
                    highest = HIGHEST.load(Ordering::Relaxed),
                    count = self.count,
                    for_draw,
                    times = n,
                    "title slot past the host heap"
                );
            }
            return Ok(None);
        }
        let header = header_at(handle);
        let [tag, i, generation] = header;
        // Generations start at 1, so a made slot's key is never 0.
        let key = u64::from(generation) << 32 | u64::from(i);
        let view = (tag == VIEW_TAG).then(|| view_at(i, generation)).flatten();
        if !for_draw {
            // The title wrote the slot: what shaders that pick textures by
            // index may find there.
            self.queue_texture(index, view.as_ref().map(|v| (key, v)));
        }
        // A texture the title placed itself is viewed (shader or unordered
        // access) through the resource made for its memory, which another
        // replaces when a texture the GPU writes comes there (#643): the
        // record stays the same, so the slot is made again when the host
        // texture changes. Resolved before the lock, since making the
        // resource can wait.
        let (placed, building) = match &view {
            Some(v @ View::Placed(_, resource, _)) if for_draw && resource[0] != 1 => {
                let (made, building) = placed_texture_or_building(v);
                (Some(made), building)
            }
            _ => (None, false),
        };
        // A texture object's host copy is there for the GPU only once it is
        // resident, which may be after the title wrote its views: the slot,
        // null until then, is made again once there is one (another host
        // copy, the same).
        let host_of =
            |r: usize| host_copy_of(r as crate::object::This).map_or(0, |t| t.as_raw() as usize);
        let texture = match (&placed, &view) {
            (Some(p), _) => Some(match p {
                Some(View::Shader(r, _) | View::Unordered(r, ..)) => (host_of(*r), *r),
                _ => (0, 0),
            }),
            (None, Some(View::Shader(r, _) | View::Unordered(r, _, _)))
                if crate::resource::buffer_bytes(*r as crate::object::This).is_none() =>
            {
                Some((host_of(*r), 0))
            }
            _ => None,
        };
        // Whether the slot is to be made again: another record, or another
        // host texture. The host texture held by the slot cannot be
        // another's (`kept` holds it), so the same address is the same one.
        let stale = |made: &Made| {
            made.keys[index as usize] != key
                || texture.is_some_and(|t| made.textures.get(&index).copied() != Some(t))
        };
        // A buffer view's host copy, a resource (which can wait in the
        // kernel, on video memory), is made before the lock too when the
        // slot is to be made: under the lock, threads recording other lists
        // only wait for descriptors to be written (VidMm waits there took
        // nearly half the title's recording time in the world).
        let resolved = placed.as_ref().and_then(Option::as_ref).or(view.as_ref());
        if for_draw
            && let Some((address, len)) = resolved.and_then(copy_range)
            && stale(&self.made.lock().unwrap_or_else(|e| e.into_inner()))
        {
            if copies {
                let _ = crate::shadow::find(address, len);
            } else if crate::shadow::existing(address, len).is_none() {
                return Err(Unmade::NoCopy);
            }
        }
        #[cfg(test)]
        if let Some(between) = BETWEEN.with(|b| b.borrow_mut().take()) {
            between();
        }
        let mut made = self.made.lock().unwrap_or_else(|e| e.into_inner());
        if header_at(handle) != header {
            // Published meanwhile.
            return Err(Unmade::Rewritten);
        }
        let Some(view) = view else {
            // Not a view of ours (never written, or of a resource the
            // title released): nothing to read.
            let mut old = None;
            if made.keys[index as usize] != 0 {
                made.keys[index as usize] = 0;
                made.textures.remove(&index);
                if self.sampler {
                    self.null_sampler(index);
                } else {
                    null_view(self.cpu(index));
                }
                self.forget_view(index);
                made.unread(index);
                old = self.swap_kept(index, None);
            }
            // A host resource the slot held goes after the lock.
            drop(made);
            drop(old);
            return Ok(None);
        };
        let mut old = None;
        // No host texture for a placed view now. While another thread is
        // still making it, a slot already made for this record over a
        // texture keeps it: a mirror racing the maker would otherwise leave
        // the slot null until it is made again. With no mapped page or a
        // failed creation, the slot follows the memory and is made null.
        let pending = matches!(placed, Some(None));
        let holds = |made: &Made| {
            made.keys[index as usize] == key
                && made.textures.get(&index).is_some_and(|&(_, r)| r != 0)
        };
        if stale(&made) && !(pending && building && holds(&made)) {
            let view = placed.as_ref().and_then(Option::as_ref).unwrap_or(&view);
            let later = !for_draw && view.is_buffer();
            made.keys[index as usize] = if later { 0 } else { key };
            // Made null until a draw names it, or a shader that picks slots
            // at run time is bound (`picked_spans`): buffer views only. A
            // view of a texture the title placed itself would make a host
            // texture over memory the title may be drawing a target in, and
            // constant buffer views are no buffer a raw read can take.
            let picking = run_time_views();
            if later && picking && buffer_range(view).is_some() {
                made.wait(index);
            }
            match texture {
                Some(t) if !later => made.textures.insert(index, t),
                _ => made.textures.remove(&index),
            };
            let mut viewed = None;
            let kept = if later || pending {
                None
            } else {
                let wanted = (self.strides.is_some() || picking).then_some(&mut viewed);
                make_with(view, self.cpu(index), wanted)
            };
            if kept.is_none() && !self.sampler {
                null_for(view, self.cpu(index));
            }
            // The entry with the descriptor, under the same lock.
            let noted = self.note_view(index, viewed.as_ref());
            if noted == Noted::Refused {
                // No raw view for the table now: the slot is made again by
                // the next draw that reads it, or the next sweep, which
                // finds one once the GPU has given one back.
                made.keys[index as usize] = 0;
                if picking {
                    made.wait(index);
                }
            }
            // What a shader that picks the slot at run time reads: the
            // bytes the slot's view or the table's raw view reads, as cut
            // to the host copy (a view the host leaves null may have the
            // latter).
            let read = viewed
                .as_ref()
                .filter(|v| picking && v.len > 0 && (kept.is_some() || noted == Noted::Named));
            match read {
                Some(v) => made.read(index, v),
                None => made.unread(index),
            }
            old = self.swap_kept(index, kept.flatten());
        }
        drop(made);
        drop(old);
        Ok(Some(placed.flatten().unwrap_or(view)))
    }

    /// The resource made for the memory of the texture the title placed
    /// itself that slot `index` was last made from for a draw, if it is
    /// one: an address only, which may have been released since.
    pub(crate) fn placed_resource(&self, index: u32) -> Option<usize> {
        let made = self.made.lock().unwrap_or_else(|e| e.into_inner());
        made.textures
            .get(&index)
            .map(|&(_, resource)| resource)
            .filter(|&r| r != 0)
    }

    /// The view host slot `index` was made from, in a few words, with the
    /// title and host addresses of a buffer view (diagnostics). "null" for
    /// a slot never made or left null.
    #[cfg(feature = "diagnostics")]
    pub(crate) fn describe_slot(&self, index: u32) -> Option<String> {
        let key = *self
            .made
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .keys
            .get(index as usize)?;
        if key == 0 {
            return Some("null".into());
        }
        let view = view_at(key as u32, (key >> 32) as u32)?;
        let mut text: String = format!("{view:?}").chars().take(120).collect();
        if let Some((address, len)) = buffer_range(&view) {
            let host = crate::shadow::find(address, len.max(1)).map(|s| s.gpu(address));
            text.push_str(&format!(" title {address:#x}+{len:#x} host {host:#x?}"));
        }
        Some(text)
    }
}

/// Tries of [`HostHeap::mirror`] at a slot published meanwhile each time.
const MIRROR_TRIES: u32 = 4;

/// Why [`HostHeap::mirror_once`] made nothing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Unmade {
    /// Another thread published a record in the slot meanwhile.
    Rewritten,
    /// The view's host copy is not made yet, and this try may not make it.
    NoCopy,
}

/// The record header the title has at `handle`: tag, view index and
/// generation, read under the lock of every record access
/// ([`super::header`]: never half of one record and half of another).
fn header_at(handle: u64) -> [u32; 3] {
    super::header(handle)
}

#[cfg(test)]
thread_local! {
    /// Run once by the next [`HostHeap::mirror`] on this thread, between
    /// reading the slot's record and taking the slot lock (tests: another
    /// thread publishing a record then).
    static BETWEEN: std::cell::RefCell<Option<Box<dyn FnOnce()>>> =
        const { std::cell::RefCell::new(None) };
}

/// A view description from its words, with the format as the host has it
/// (a typed buffer view's keeps the title's element size).
pub(crate) fn srv_desc(words: &[u32; 10]) -> D3D12_SHADER_RESOURCE_VIEW_DESC {
    let mut words = *words;
    words[0] = crate::resource::pc_view_format(&words, false);
    // SAFETY: 40 bytes in the host's layout.
    unsafe {
        std::ptr::from_ref(&words)
            .cast::<D3D12_SHADER_RESOURCE_VIEW_DESC>()
            .read_unaligned()
    }
}

pub(crate) fn uav_desc(words: &[u32; 10]) -> D3D12_UNORDERED_ACCESS_VIEW_DESC {
    let mut words = *words;
    words[0] = crate::resource::pc_view_format(&words, true);
    // SAFETY: 40 bytes in the host's layout.
    unsafe {
        std::ptr::from_ref(&words)
            .cast::<D3D12_UNORDERED_ACCESS_VIEW_DESC>()
            .read_unaligned()
    }
}

/// A buffer view of title memory at `address` on its host copy: the first
/// element moves by the copy's offset.
/// Bytes per element of a buffer view.
fn element_size(desc: &D3D12_SHADER_RESOURCE_VIEW_DESC) -> u64 {
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { desc.Anonymous.Buffer };
    if buffer.StructureByteStride > 0 {
        u64::from(buffer.StructureByteStride)
    } else if buffer.Flags.0 & D3D12_BUFFER_SRV_FLAG_RAW.0 != 0 {
        4
    } else {
        crate::resource::element(desc.Format.0 as u32).map_or(0, |(bytes, _)| bytes)
    }
}

/// The title memory a buffer shader or unordered access view reads: its
/// address and length.
pub(crate) fn buffer_range(view: &View) -> Option<(u64, u64)> {
    let (base, words): (u64, &[u32; 10]) = match view {
        View::Shader(r, Some(desc)) => (
            crate::resource::buffer_bytes(*r as crate::object::This)?.0 as u64,
            desc,
        ),
        View::Placed(false, resource, desc) if resource[0] == 1 => (
            u64::from(desc[10]) | u64::from(desc[11]) << 32,
            desc[..10].try_into().ok()?,
        ),
        View::Unordered(r, _, Some(desc)) => {
            let base = crate::resource::buffer_bytes(*r as crate::object::This)?.0 as u64;
            return unordered_range(base, desc);
        }
        View::Placed(true, resource, desc) if resource[0] == 1 => {
            let base = u64::from(desc[10]) | u64::from(desc[11]) << 32;
            return unordered_range(base, desc[..10].try_into().ok()?);
        }
        _ => return None,
    };
    let mut desc = srv_desc(words);
    desc.Format = DXGI_FORMAT(words[0] as i32);
    let element = element_size(&desc);
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { desc.Anonymous.Buffer };
    range_of(base, buffer.FirstElement, buffer.NumElements, element)
}

/// The title memory of `elements` elements of `element` bytes from element
/// `first` of memory at `base`: none if it cannot be represented (a title
/// description out of range, read from an `extern "system"` call, where an
/// overflow would end the process or wrap around).
fn range_of(base: u64, first: u64, elements: u32, element: u64) -> Option<(u64, u64)> {
    let start = base.checked_add(first.checked_mul(element)?)?;
    let len = u64::from(elements).checked_mul(element)?;
    start.checked_add(len)?;
    Some((start, len))
}

/// [`buffer_range`] of an unordered access view of memory at `base`.
fn unordered_range(base: u64, words: &[u32; 10]) -> Option<(u64, u64)> {
    let desc = uav_desc(words);
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { desc.Anonymous.Buffer };
    let element = if buffer.StructureByteStride > 0 {
        u64::from(buffer.StructureByteStride)
    } else if buffer.Flags.0 & D3D12_BUFFER_UAV_FLAG_RAW.0 != 0 {
        4
    } else {
        crate::resource::element(words[0])?.0
    };
    range_of(base, buffer.FirstElement, buffer.NumElements, element)
}

impl HostHeap {
    /// The runtime stride table's GPU address, and its buffer (for lists
    /// to keep), if the heap has one.
    pub(crate) fn stride_table(&self) -> Option<(u64, &ID3D12Resource)> {
        self.strides
            .as_ref()
            .map(|s| (s.table.gpu, &s.table.resource))
    }

    /// A note for a recording that reads the stride table ([`StrideHold`]);
    /// none if the heap has no table. The raw views the GPU is done with go
    /// back to the pool now (as they do whenever one is taken or retires,
    /// and when the host fence reaches them: [`fence_reached`]).
    pub(crate) fn stride_hold(&self) -> Option<StrideHold> {
        let s = self.strides.as_ref()?;
        s.with_pool(|s, a| s.reap(a));
        Some(StrideHold)
    }

    /// Slot `index`'s table entry for the view just made there: `viewed`,
    /// the bytes of its host copy a buffer view reads (none for any other
    /// view, or one the host could not make). See [`Strides::name`].
    fn note_view(&self, index: u32, viewed: Option<&Viewed>) -> Noted {
        match &self.strides {
            Some(s) => s.note(index, viewed),
            None => Noted::Nothing,
        }
    }

    /// Slot `index` holds no view shaders read through the stride table:
    /// its entry says none, then its raw view retires.
    fn forget_view(&self, index: u32) {
        if let Some(s) = &self.strides {
            s.note(index, None);
        }
    }

    /// [`HostHeap::forget_view`] for each slot of `ranges` (dynamic slots
    /// given back), under one lock.
    fn forget_ranges(&self, ranges: &[(u32, u32)]) {
        let Some(s) = &self.strides else { return };
        s.with_pool(|s, a| {
            let slots = || {
                ranges
                    .iter()
                    .flat_map(|&(first, n)| first..first.saturating_add(n).min(s.alias_base))
            };
            for index in slots() {
                a.unwait(index);
                s.set_entry(index, StrideEntry::NONE);
            }
            if a.of.is_empty() {
                return;
            }
            // After every entry says none: the fence value read when one
            // retires counts them.
            for index in slots() {
                if let Some(old) = a.of.remove(&index) {
                    a.unname(old);
                }
            }
            s.reap(a);
        });
    }

    /// What the buffer views of the title's slots read and write, for a
    /// shader that picks the slots it reads when it runs (no used-resource
    /// entry names them, so no draw's table made their views: they were
    /// null views, read as zeros; VIctory #71, watch.md item 170), with the
    /// buffer views the title wrote since the last call made first. `start`
    /// is the title heap's first descriptor. `None` without
    /// `XTP_RUN_TIME_VIEWS=1` ([`run_time_views`]).
    ///
    /// Every draw or dispatch that binds such a shader calls it (`draw.rs`
    /// `picked_buffers`), and so does each list that did as it closes and
    /// at each of its submissions; each list's step ([`PickedStep`]) has
    /// what it lists refreshed for its own submission, once per submission
    /// whichever lists ask: whichever list the title submits first, or
    /// drops, each reads the title memory of its own submission (a sweep
    /// once a frame for the frame's first list left the others the frame
    /// before's, and views written after it null).
    ///
    /// Every view waiting is made, never left null for a later call (Codex
    /// review of 244cf78, finding 6): those whose host copy is there first
    /// (a descriptor or two), then those whose host copy is to be made, a
    /// resource each, which can wait in the kernel: past [`PICKED_BUDGET`]
    /// in a frame it is logged, not put off (it was, and a dispatch that
    /// picked a view left null read 0). A view the title writes after a
    /// list's last such draw is made as the list closes, or, written after
    /// that, as the list is submitted (Codex review of a090f22, point 6: a
    /// view written between the close and the submission stayed null), and
    /// its bytes refreshed for the submission. So the work of a call is
    /// bounded by what the title wrote since the last: the queue holds each
    /// view once, and nothing waits from one call to the next; the listing
    /// is kept as views change ([`Listing`]), and a call adds the list
    /// nothing but its step for the refresh epoch ([`PickedUse::defer`]),
    /// the runs being given at submission ([`Listing::due`]).
    ///
    /// The approach of Marvisak/VIctory#1544 (n-romanovich): views written
    /// by the title made before a draw that reads the heap directly, their
    /// title memory brought up to date. Here only the slots that waited are
    /// made (not every slot that ever held a view, at each change of the
    /// heap), the spans are merged per host copy (fxceless's VIctory fork,
    /// 578d459, refreshed each view's memory per list: 18 s a frame), and
    /// the views go through the checks a host view must pass (`fit_view`),
    /// so that a bad record makes a null view rather than remove the
    /// device.
    pub(crate) fn picked_spans(&self, start: u64) -> Option<Arc<Picked>> {
        if !run_time_views() || self.sampler {
            return None;
        }
        let frame = crate::frame::current();
        let started = std::time::Instant::now();
        let mut sweeps = self.picked.lock().unwrap_or_else(|e| e.into_inner());
        // A frame's time, or [`PASS_AGAIN`]'s where frames do not advance.
        let again = sweeps
            .began
            .is_none_or(|at| started.saturating_duration_since(at) >= PASS_AGAIN);
        if sweeps.frame != frame || again {
            sweeps.frame = frame;
            sweeps.began = Some(started);
            sweeps.spent = std::time::Duration::ZERO;
        }
        let waiting = {
            let mut made = self.made.lock().unwrap_or_else(|e| e.into_inner());
            let waiting = std::mem::take(&mut made.waiting);
            for &index in &waiting {
                if let Some(queued) = made.queued.get_mut(index as usize) {
                    *queued = false;
                }
            }
            waiting
        };
        // Those whose host copy is there first, then those that need one,
        // in the order the title wrote them: every one.
        let mut left = Vec::new();
        for &index in &waiting {
            let handle = start + u64::from(index) * 32;
            if self.mirror_tries(index, handle, false) == Err(Unmade::NoCopy) {
                left.push(index);
            }
        }
        let copying = std::time::Instant::now();
        for &index in &left {
            let _ = self.mirror_tries(index, start + u64::from(index) * 32, true);
        }
        let before = sweeps.spent;
        sweeps.spent += copying.elapsed();
        let over = before < PICKED_BUDGET && sweeps.spent >= PICKED_BUDGET;
        static PASSES: AtomicU64 = AtomicU64::new(0);
        static OVER: AtomicU64 = AtomicU64::new(0);
        let passes = PASSES.fetch_add(1, Ordering::Relaxed) + 1;
        let overs = if over {
            OVER.fetch_add(1, Ordering::Relaxed) + 1
        } else {
            0
        };
        if passes.is_power_of_two() || overs.is_power_of_two() {
            let views = self
                .made
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .buffers
                .len();
            let (copies, runs, bytes, hull, written) = self.listed.listing().census();
            let [refreshes, refreshed, declared] =
                PICKS.each_ref().map(|n| n.load(Ordering::Relaxed));
            tracing::info!(
                passes,
                frames_past_budget = OVER.load(Ordering::Relaxed),
                waited = waiting.len(),
                copies_made = left.len(),
                views,
                copies,
                runs,
                written,
                mib = bytes >> 20,
                hull_mib = hull >> 20,
                refreshes,
                refreshed_mib = refreshed >> 20,
                declared,
                us = started.elapsed().as_micros() as u64,
                frame_copy_us = sweeps.spent.as_micros() as u64,
                "buffer views of slots picked at run time: made and listed"
            );
        }
        Some(self.listed.clone())
    }

    /// [`HostHeap::mirror`] for a draw, `copies` as for
    /// [`HostHeap::mirror_once`].
    fn mirror_tries(&self, index: u32, handle: u64, copies: bool) -> Result<(), Unmade> {
        for _ in 0..MIRROR_TRIES {
            match self.mirror_once(index, handle, true, copies) {
                Ok(_) => return Ok(()),
                Err(Unmade::Rewritten) => continue,
                Err(Unmade::NoCopy) => return Err(Unmade::NoCopy),
            }
        }
        Err(Unmade::Rewritten)
    }

    /// Host copies of the listing that recordings catching up looked at
    /// so far ([`Listing::since`]; tests).
    #[cfg(test)]
    pub(crate) fn visits(&self) -> u64 {
        self.listed.listing().visits
    }

    /// Title slots whose buffer view waits for a sweep (tests).
    #[cfg(test)]
    pub(crate) fn waiting(&self) -> usize {
        self.made
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .waiting
            .len()
    }

    /// Starts the frame's time for [`HostHeap::picked_spans`] afresh, as a
    /// new frame does (tests: the frame count is the process's, and other
    /// tests count it).
    #[cfg(test)]
    pub(crate) fn forget_picked_pass(&self) {
        self.picked.lock().unwrap_or_else(|e| e.into_inner()).spent = std::time::Duration::ZERO;
    }

    /// Spends the frame's time for [`HostHeap::picked_spans`]'s host
    /// copies, as a first copy that took it all would (tests).
    #[cfg(test)]
    fn spend_picked_time(&self) {
        let mut sweeps = self.picked.lock().unwrap_or_else(|e| e.into_inner());
        sweeps.frame = crate::frame::current();
        sweeps.began = Some(std::time::Instant::now());
        sweeps.spent = PICKED_BUDGET;
    }

    /// Slot `index`'s table entry (tests).
    #[cfg(test)]
    pub(crate) fn entry(&self, index: u32) -> StrideEntry {
        let s = self.strides.as_ref().expect("a stride table");
        assert!(index < s.table.len);
        // SAFETY: as in `set_entry`.
        let at = unsafe {
            AtomicU64::from_ptr((s.table.ptr as *mut u64).add((index + HEADER) as usize))
        };
        StrideEntry::from_bits(at.load(Ordering::Acquire))
    }

    /// The table's header words: title slots mirrored, entries (tests).
    #[cfg(test)]
    fn table_header(&self) -> (u32, u32) {
        let s = self.strides.as_ref().expect("a stride table");
        // SAFETY: the header, inside the live mapping.
        let words = unsafe { (s.table.ptr as *const [u64; 2]).read() };
        ((words[0] >> 32) as u32, (words[1] >> 32) as u32)
    }

    /// The first slot of the raw views' pool (tests).
    #[cfg(test)]
    pub(crate) fn alias_base(&self) -> u32 {
        self.strides.as_ref().expect("a stride table").alias_base
    }

    /// Pool slots retiring, and free or never taken (tests).
    #[cfg(test)]
    fn pool_state(&self) -> (usize, u32) {
        let s = self.strides.as_ref().expect("a stride table");
        let a = s.aliases.lock().unwrap_or_else(|e| e.into_inner());
        (a.retiring.len(), a.free.len() as u32 + (a.size - a.next))
    }
}

/// Frames for which a slot a draw read through a table counts as a table's
/// ([`HostHeap::texture_slot_tabled`]): the title's writes of it wait for
/// such a draw ([`TABLED_GRACE`]).
const TABLED_FRAMES: u64 = 240;

/// Frames for which passes leave a served slot a draw read through a table
/// to such draws, which bring it up to date: this one and the one before.
/// Not [`TABLED_FRAMES`]: a title that stops reading a slot through tables
/// to pick it by index would see the image it had for 240 frames.
const TABLED_CHECKS: u64 = 1;

/// Frames a texture view the title writes into a table's slot waits in the
/// [`TextureQueue`] for the draw that reads it, before a pass takes it for
/// one a shader picks by index. A title that builds a table per draw
/// writes thousands of texture views a frame into slots only tables read
/// (fxceless's VIctory fork counted some 2,500 a frame in BTP 6's opening
/// cutscene, a few hundred textures over and over), each of which would
/// stand in the queue with the slots shaders do pick by index. In frames
/// as `crate::frame::current` counts them (two or three begun per frame
/// presented in that cutscene, by the fork's count).
const TABLED_GRACE: u64 = 8;

/// Since the start: texture views the title wrote into a table's slot
/// (due after [`TABLED_GRACE`]), and slots that stopped waiting for a pass
/// because a draw read them through a table.
static TABLED_WRITES: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// Slots of each kind a pass is handed at most: due for their first
/// upload, served to check for changes.
const PASS_SLOTS: usize = 4096;

/// Where frames do not advance (a title that does not wait for them), a
/// pass this long after the last one.
const PASS_AGAIN: std::time::Duration = std::time::Duration::from_millis(100);

/// How long a pass makes and uploads textures for the first time: making
/// one can wait in the kernel (a host texture, its staging buffer), and
/// the world's loading leaves thousands waiting; the rest wait for the
/// next frames' passes. (The fork's 250 ms a pass came after "1.6-2.2 s of
/// each of the world's first twenty frames" without a limit.)
pub(crate) const PASS_MAKING: std::time::Duration = std::time::Duration::from_millis(8);

/// How long a pass checks served textures for changes, in turns from
/// where the last one stopped.
pub(crate) const PASS_CHECKING: std::time::Duration = std::time::Duration::from_millis(1);

/// The bytes of served textures whose memory only a hash tells (neither
/// write-watched nor guarded: `texture::Change::Unknown`) a pass uploads
/// for the check at submission to hash: some 1.3 ms of hashing at the
/// safety net's measured 3 GB/s (`texture.rs` `net_costs`). The others
/// wait for their next turn.
pub(crate) const PASS_HASHING: u64 = 4 << 20;

/// The bytes of staging buffers that live past which a pass uploads no
/// texture for the first time (tables' textures counted): each one is
/// upload memory of its texture's size, and the world's heap holds views
/// of some 32,000 textures. Those past it wait (logged once). It was
/// 3 GiB: reached in the first world session (heaptex-A, PC, 2026-11-25),
/// beside the title's own memory, it left the PC's memory nearly full.
/// Staging buffers were then held for their texture's whole life, so this
/// one filled for good as the world loaded (imgsrc, §11.2 of
/// `docs/analysis/pix-world-frame.md`: 14,000 to 19,000 slots waiting,
/// null); now each goes once its last copy ran and no upload of it was
/// recorded for [`RELEASE_IDLE`] passes (`texture::release_staging`), and
/// is made again for its next upload.
pub(crate) const PASS_STAGING: u64 = 1 << 30;

/// Passes without an upload recorded after which a staging buffer no list
/// holds goes: a texture sampled at every frame keeps its own.
const RELEASE_IDLE: u64 = 2;

/// Textures a pass looks at for staging buffers to release, in turns from
/// where the last one stopped (the world has some 30,000).
const RELEASE_LOOKS: usize = 4096;

/// How many times its base budget of bytes (`XTP_HEAP_TEXTURES`) a pass
/// may stage while the one before left due slots for later passes: a
/// backlog, as when the world loads (thousands of slots waiting). Within
/// the staging limit and the video memory share all the same, each texture
/// checked with its bytes; back to the base once a pass leaves none.
pub(crate) const PASS_LOADING: u64 = 4;

/// The most `XTP_PASS_STAGING` may set: 4 GiB of upload memory beside the
/// title's own already takes a large share of the PC's memory.
const PASS_STAGING_MOST: u64 = 4 << 30;

/// The staging limit in force: [`PASS_STAGING`], or `XTP_PASS_STAGING`
/// (MiB, at most [`PASS_STAGING_MOST`]) for an A/B of the limit. Read once.
pub(crate) fn pass_staging() -> u64 {
    static LIMIT: OnceLock<u64> = OnceLock::new();
    *LIMIT.get_or_init(|| {
        let limit = pass_staging_of(std::env::var("XTP_PASS_STAGING").ok().as_deref());
        if limit != PASS_STAGING {
            tracing::warn!(
                mib = limit >> 20,
                "XTP_PASS_STAGING: staging limit of the textures picked by index"
            );
        }
        limit
    })
}

/// [`pass_staging`] for the switch's value: a number of MiB, at most
/// [`PASS_STAGING_MOST`]; anything else (or none) gives [`PASS_STAGING`].
fn pass_staging_of(value: Option<&str>) -> u64 {
    match value.map(str::trim).map(str::parse::<u64>) {
        // Bounded in MiB before the shift: from 2^44 MiB on, it wrapped to
        // 0 (Codex's review of c44a716, 5).
        Some(Ok(mib)) if mib > 0 => mib.min(PASS_STAGING_MOST >> 20) << 20,
        Some(_) => {
            tracing::warn!("XTP_PASS_STAGING is not a number of MiB: the default limit");
            PASS_STAGING
        }
        None => PASS_STAGING,
    }
}

/// The share of its local video memory budget (percent) past which a pass
/// makes no host texture for a first upload: in heaptex-A the passes made
/// one for each of 17,500 slots, the process's local memory went from the
/// 10.5 GB of the same world without them to 13.3 GB resident, past what
/// a 16 GB card shared with the desktop's programs has, and the video
/// memory manager's evictions at each submission's preparation took the
/// frame to 1-11 s on the GPU. Those past it wait for room (the title's
/// own textures and targets come first). Checked per texture, with its
/// bytes: one 64 MiB texture does not pass because 1 MiB is left.
pub(crate) const PASS_VIDEO_SHARE: u64 = 85;

/// The bytes of host textures a pass may make, by the process's local
/// video memory `(budget, usage)` (`crate::dxgi::video_memory`, asked once
/// a pass): up to [`PASS_VIDEO_SHARE`] of the budget (`None`, unknown: no
/// limit).
pub(crate) fn video_room(memory: Option<(u64, u64)>) -> u64 {
    memory.map_or(u64::MAX, |(budget, usage)| {
        (budget / 100 * PASS_VIDEO_SHARE).saturating_sub(usage)
    })
}

/// The bytes of staging buffers a pass may make for its next texture, with
/// `staged` bytes of them there are and `spent` of its `budget` spent by
/// the pass so far: within the staging limit ([`pass_staging`]) and the
/// budget; the first of a pass may take more than the budget (it would
/// never fit otherwise).
pub(crate) fn staging_room(staged: u64, spent: u64, budget: u64) -> u64 {
    staging_room_within(pass_staging(), staged, spent, budget)
}

/// [`staging_room`] under the staging limit `limit`.
fn staging_room_within(limit: u64, staged: u64, spent: u64, budget: u64) -> u64 {
    let pass = if spent == 0 {
        u64::MAX
    } else {
        budget.saturating_sub(spent)
    };
    limit.saturating_sub(staged).min(pass)
}

/// Makes the staging buffer of `resource` (host copy `texture`) for an
/// upload by a pass over the textures shaders pick by index (`draw.rs`
/// `heap_textures`) if it has none, within the room left
/// ([`staging_room`]: `spent` bytes of the pass's `budget` so far, counted
/// here): what came of it. Staging memory too full for it counts the pass
/// as full in `counts` (logged once).
pub(crate) fn stage_for_pass(
    resource: crate::object::This,
    texture: &ID3D12Resource,
    budget: u64,
    spent: &mut u64,
    counts: &mut PassCounts,
) -> crate::texture::Staged {
    use crate::texture::Staged;
    let staged = crate::texture::staged_total();
    let made = crate::texture::stage(resource, texture, staging_room(staged, *spent, budget));
    match made {
        Staged::Ready(bytes) => *spent += bytes,
        Staged::NoRoom(bytes) if bytes > pass_staging().saturating_sub(staged) => {
            counts.full = 1;
            once!(
                mib = staged >> 20,
                "textures shaders pick by index: staging memory full, the rest wait"
            );
        }
        _ => {}
    }
    made
}

/// The bytes a frame's pass stages for first uploads of textures shaders
/// pick by index: the base budget ([`heap_textures_base`]), or
/// [`PASS_LOADING`] times it while the passes leave due slots (a backlog).
/// `None`: no pass.
pub(crate) fn heap_textures_budget() -> Option<u64> {
    heap_textures_base().map(|base| pass_budget(base, LOADING.load(Ordering::Relaxed)))
}

/// The base budget of a frame's pass (`XTP_HEAP_TEXTURES`, MiB, 64 by
/// default: the detiling at submission is about a millisecond per MiB);
/// `None` when it is 0, `off` or `false`: no pass, as before (such
/// textures stay as they were made, and placed ones null). Read once.
fn heap_textures_base() -> Option<u64> {
    static BUDGET: OnceLock<Option<u64>> = OnceLock::new();
    *BUDGET
        .get_or_init(|| heap_textures_budget_of(std::env::var("XTP_HEAP_TEXTURES").ok().as_deref()))
}

/// Whether the last pass left due slots for later passes ([`PASS_LOADING`]).
/// One for the process: the title's view heap is the one with passes.
static LOADING: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// The bytes a pass may stage: `base`, or [`PASS_LOADING`] times it while
/// `loading`.
fn pass_budget(base: u64, loading: bool) -> u64 {
    if loading {
        base.saturating_mul(PASS_LOADING)
    } else {
        base
    }
}

/// Whether the pass after one that left `left` due slots for later passes
/// (by its budgets of bytes and time, staging or video memory) may spend
/// more: slots still wait (a pass raised to more that leaves none goes
/// back to the base). Before, it was whether the pass spent all of its
/// base: one whose next texture did not fit in what was left (six 10 MiB
/// staging buffers of 64 MiB, the seventh refused) kept the next at the
/// base, slots waiting (Codex's review of c44a716, 4).
fn loading_after(left: u64) -> bool {
    left > 0
}

/// [`heap_textures_budget`] for the switch's value.
fn heap_textures_budget_of(value: Option<&str>) -> Option<u64> {
    const DEFAULT_MIB: u64 = 64;
    let mib = match value.map(str::trim) {
        None => DEFAULT_MIB,
        Some(v) if matches!(v.to_ascii_lowercase().as_str(), "0" | "off" | "false") => {
            tracing::warn!(
                "XTP_HEAP_TEXTURES={v}: textures shaders pick by index are not uploaded"
            );
            return None;
        }
        Some(v) => v.parse::<u64>().unwrap_or_else(|_| {
            tracing::warn!(
                value = v,
                "XTP_HEAP_TEXTURES is not a number of MiB: {DEFAULT_MIB} MiB"
            );
            DEFAULT_MIB
        }),
    };
    Some(mib.min(1 << 20) << 20)
}

/// What the queue of textures picked by index knows of a title slot that
/// waits for its first upload.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Waiting {
    /// The frame from which it is due.
    due: u64,
    /// The frame it was written in.
    written: u64,
    /// Which write of the queue put it here, counted over the run: newer
    /// writes are served first, and the pass hands it back with what it
    /// did, so that a slot written again meanwhile stays as written.
    write: u64,
    /// The record it holds (view index and generation).
    key: u64,
    /// Looks in a row that found nothing to upload (none of its memory
    /// mapped, its host copy not resident yet).
    looks: u8,
}

/// The title slots of a view heap holding a view of a texture the title
/// may fill itself (a texture object it does not render to, a texture
/// placed in its memory), for shaders that pick textures by an index they
/// compute: no used-resource entry names such a slot, so no draw's table
/// makes it (a placed texture's slot stays null) nor uploads its texture
/// (the host copy keeps what it was made with). A pass once a frame
/// (`draw.rs` `heap_textures`) serves the slots waiting for their first
/// upload, then checks the served ones for changes in turns. A slot a draw
/// reads through a table is that draw's to bring up to date. No host heap
/// is needed, so it is tested without one.
///
/// Ported from fxceless's VIctory fork (commit 578d459), `TextureQueue`:
/// its order (the slots looked at before, then the newest write and the
/// oldest in turns), its looks again at a slot with nothing to upload yet,
/// and its grace for the slots of tables. The fork uploads a texture once,
/// and again only when the title rewrites its slot or the runtime maps
/// its memory; here an uploaded slot stays served and is checked against
/// the write watch or its guards (`texture::memory_change`: a texture
/// streaming in is written after its view), which takes the place of the
/// fork's mapping hooks, spans of placed memory and part uploads. Slots
/// are kept by write as well as by slot: the fork sorted every due slot
/// at every pass (tens of thousands while the world loads).
#[derive(Default)]
pub(crate) struct TextureQueue {
    /// Slots waiting for their first upload.
    waiting: BTreeMap<u32, Waiting>,
    /// The same by write: never looked at, and looked at before.
    fresh: BTreeMap<u64, u32>,
    again: BTreeMap<u64, u32>,
    /// Writes so far.
    writes: u64,
    /// Slots uploaded, with the record they held then.
    served: BTreeMap<u32, u64>,
    /// The served slot the last checks stopped at.
    cursor: u32,
    /// The last pass handed out: its frame and when.
    pass: Option<(u64, std::time::Instant)>,
}

impl TextureQueue {
    /// Slot `index` was written with record `key`'s view of a texture the
    /// title may fill (`placed` in its memory) in `frame`, due from frame
    /// `due`: it waits for its first upload. Not if it holds the record it
    /// waits with, or that of a texture object it was served with (a title
    /// copying the same descriptors again): the host slot of a placed
    /// texture is made null by the write (`HostHeap::mirror`), so such a
    /// slot waits again. Whether it now waits anew.
    fn written(&mut self, index: u32, key: u64, placed: bool, frame: u64, due: u64) -> bool {
        let same = self.waiting.get(&index).is_some_and(|w| w.key == key)
            || (!placed && self.served.get(&index) == Some(&key));
        if same {
            return false;
        }
        self.remove(index);
        self.writes += 1;
        let write = self.writes;
        self.waiting.insert(
            index,
            Waiting {
                due,
                written: frame,
                write,
                key,
                looks: 0,
            },
        );
        self.fresh.insert(write, index);
        true
    }

    /// Slot `index` holds no view of a texture the title may fill: out.
    fn remove(&mut self, index: u32) {
        if let Some(w) = self.waiting.remove(&index) {
            self.fresh.remove(&w.write);
            self.again.remove(&w.write);
        }
        self.served.remove(&index);
    }

    /// Slot `index`'s texture was brought up to date (by a pass, or by a
    /// draw that read it through a table): served, if it waited. Whether
    /// it did.
    fn serve(&mut self, index: u32) -> bool {
        let Some(w) = self.waiting.remove(&index) else {
            return false;
        };
        self.fresh.remove(&w.write);
        self.again.remove(&w.write);
        self.served.insert(index, w.key);
        true
    }

    /// Up to `max` slots due at `frame`, each with its write number, in the
    /// order to serve them: first those looked at before, the newest first
    /// (the look is a query, and what arrived since is what the title
    /// streamed last); then the newest write and the oldest in turns. Slot
    /// order served the low slots of a backlog of 25,000-40,000 while the
    /// textures of the shot on screen waited (the fork's run 52: "98 % of
    /// the drawn pixels had no colour texture in the first 3D frames");
    /// the newest alone would leave the oldest waiting for good.
    fn due(&self, frame: u64, max: usize) -> Vec<(u32, u64)> {
        let is_due =
            |&(_, &index): &(&u64, &u32)| self.waiting.get(&index).is_some_and(|w| w.due <= frame);
        let mut order: Vec<(u32, u64)> = self
            .again
            .iter()
            .rev()
            .filter(&is_due)
            .take(max)
            .map(|(&write, &index)| (index, write))
            .collect();
        let mut fresh = self.fresh.iter().filter(&is_due);
        let mut newest = true;
        while order.len() < max {
            let next = if newest {
                fresh.next_back()
            } else {
                fresh.next()
            };
            let Some((&write, &index)) = next else {
                break;
            };
            order.push((index, write));
            newest = !newest;
        }
        order
    }

    /// How many slots are due at `frame`: those [`TextureQueue::due`] hands
    /// out and those past its `max`.
    fn due_count(&self, frame: u64) -> usize {
        self.waiting.values().filter(|w| w.due <= frame).count()
    }

    /// The waiting entry of `index`, if it is still the write the pass was
    /// handed.
    fn same(&mut self, index: u32, write: u64) -> Option<&mut Waiting> {
        self.waiting.get_mut(&index).filter(|w| w.write == write)
    }

    /// Slot `index` (handed out with `write`) was uploaded: served.
    fn done(&mut self, index: u32, write: u64) {
        if self.same(index, write).is_some() {
            self.serve(index);
        }
    }

    /// Nothing to upload at slot `index` (handed out with `write`) now:
    /// looked at again after 1, 2, 4, ... frames, at most `most`, before
    /// the slots never looked at.
    fn later(&mut self, index: u32, write: u64, frame: u64, most: u64) {
        let Some(w) = self.same(index, write) else {
            return;
        };
        w.due = frame + (1u64 << w.looks.min(16)).min(most.max(1));
        w.looks = w.looks.saturating_add(1);
        if self.fresh.remove(&write).is_some() {
            self.again.insert(write, index);
        }
    }

    /// Slot `index` holds nothing the title fills (its view went, or views
    /// a texture the host renders to or copied into): out until written
    /// again. `write`: as the pass was handed it; `None`, a served slot.
    fn dropped(&mut self, index: u32, write: Option<u64>) {
        match write {
            Some(write) => {
                if self.same(index, write).is_some() {
                    self.remove(index);
                }
            }
            None => {
                self.served.remove(&index);
            }
        }
    }

    /// Up to `max` served slots after the one the last checks stopped at,
    /// going round once at most, those `skip` says to leave out.
    fn next_served(&self, max: usize, skip: impl Fn(u32) -> bool) -> Vec<u32> {
        use std::ops::Bound::{Excluded, Unbounded};
        let after = self.served.range((Excluded(self.cursor), Unbounded));
        let before = self.served.range(..=self.cursor);
        after
            .chain(before)
            .map(|(&index, _)| index)
            .filter(|&index| !skip(index))
            .take(max)
            .collect()
    }

    /// Whether a pass is due at `frame` and `now`: once a frame, or
    /// [`PASS_AGAIN`] after the last one where frames do not advance. If
    /// it is, it is taken: the time since the last one (the longest there
    /// is for the first).
    fn pass_due(&mut self, frame: u64, now: std::time::Instant) -> Option<std::time::Duration> {
        let since = match self.pass {
            None => std::time::Duration::MAX,
            Some((last, at)) => {
                let since = now.saturating_duration_since(at);
                if last == frame && since < PASS_AGAIN {
                    return None;
                }
                since
            }
        };
        self.pass = Some((frame, now));
        Some(since)
    }

    /// For the log: slots waiting; of them due at `frame`; looked at
    /// before; slots served; and the frames the oldest due one has waited
    /// since its write.
    fn census(&self, frame: u64) -> [u64; 5] {
        let mut c = [
            self.waiting.len() as u64,
            0,
            self.again.len() as u64,
            self.served.len() as u64,
            0,
        ];
        for w in self.waiting.values().filter(|w| w.due <= frame) {
            c[1] += 1;
            c[4] = c[4].max(frame.saturating_sub(w.written));
        }
        c
    }
}

/// Whether `view`, as the title wrote it, views a texture the title may
/// fill itself: one it placed in its memory (made when something reads
/// it), or a texture object it does not render to nor write through
/// unordered access (as its host copy says, once resident).
fn fills_texture(view: &View) -> bool {
    const BUFFER: u32 = 1;
    match view {
        View::Placed(false, resource, _) => resource[0] != BUFFER,
        View::Shader(resource, _) => {
            let resource = *resource as crate::object::This;
            crate::resource::buffer_bytes(resource).is_none()
                && crate::resource::host_resource(resource)
                    .is_none_or(crate::texture::title_written)
        }
        _ => false,
    }
}

/// What a title slot the title copied a record into as bytes holds
/// ([`HostHeap::copied_kind`]), for the count of such slots and of those
/// shaders pick by index (`super::sync_heaps`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CopiedKind {
    /// A view of a texture the title fills ([`fills_texture`]): one that
    /// shaders pick by index, unless a draw read the slot through a table
    /// within [`TABLED_FRAMES`] frames (`tabled`: such draws make it).
    Texture { tabled: bool },
    /// A buffer view ([`buffer_range`]); `picked`: shaders that pick
    /// buffers at run time read the heap (its sweeps ran in this frame or
    /// the one before). The sweeps run with `XTP_RUN_TIME_VIEWS=1` only
    /// ([`run_time_views`]): without it no buffer view counts as picked,
    /// and such shaders read null views whatever the title wrote.
    Buffer { picked: bool },
    /// A slot of a sampler heap.
    Sampler,
    /// Another view of ours: a constant buffer, a texture the host renders
    /// to or writes.
    Other,
    /// No view of ours: bytes that hold no record, or a view of a resource
    /// the title released.
    None,
}

/// [`HostHeap::tabled`]'s stamp for `frame`.
fn stamp_of(frame: u64) -> u32 {
    (frame as u32).wrapping_add(1)
}

/// The frame passes and stamps count by: the title's, and in tests what
/// a test adds ([`PASS_LATER`]).
fn pass_frame() -> u64 {
    let frame = crate::frame::current();
    #[cfg(test)]
    let frame = frame + PASS_LATER.load(Ordering::Relaxed);
    frame
}

/// Frames tests add for passes: the title's frame counter is not theirs
/// to advance.
#[cfg(test)]
pub(crate) static PASS_LATER: AtomicU64 = AtomicU64::new(0);

/// What a pass over the textures shaders pick by index works on (`draw.rs`
/// `heap_textures`), from [`HostHeap::texture_pass`].
pub(crate) struct TexturePass {
    /// The queue's clock ([`HostHeap::clock`]).
    pub(crate) frame: u64,
    /// Slots due for their first upload, with their write numbers, in the
    /// order to serve them.
    pub(crate) due: Vec<(u32, u64)>,
    /// Served slots to check for changes, in turn; those tables read in
    /// the last [`TABLED_CHECKS`] frames are left out (their draws check
    /// them).
    pub(crate) served: Vec<u32>,
    /// Frames at most between two looks at a slot with nothing to upload:
    /// those in about four seconds, between 16 and 256 (a title streams by
    /// the clock; at the opening cutscene's half a frame a second, the
    /// fork's 120 frames were four minutes).
    pub(crate) most: u64,
    /// Slots due for their first upload past those in `due` (a backlog of
    /// more than [`PASS_SLOTS`]): left for later passes all the same.
    pub(crate) backlog: u64,
}

impl TexturePass {
    /// The due slots this pass leaves for later passes, `counts` being what
    /// it did: those it was handed and left, and those it was not handed.
    /// Before, only the first, and a pass that did all of the 4,096 it was
    /// handed took the next back to the base budget with more due (Codex's
    /// review of db37732, 4).
    pub(crate) fn left(&self, counts: &PassCounts) -> u64 {
        counts.left + self.backlog
    }
}

/// What one pass over the textures shaders pick by index did, for the log
/// ([`HostHeap::texture_pass_done`]).
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct PassCounts {
    /// Due slots looked at; of them uploaded for the first time, and the
    /// bytes of their staging buffers; and those with nothing to upload
    /// yet (none of the memory mapped, the host copy not resident).
    pub(crate) looked: u64,
    pub(crate) uploaded: u64,
    pub(crate) bytes: u64,
    pub(crate) away: u64,
    /// Served slots checked; of them found changed (uploaded again), and
    /// those whose memory only a hash tells, uploaded for the check at
    /// submission to hash ([`PASS_HASHING`]).
    pub(crate) checked: u64,
    pub(crate) changed: u64,
    pub(crate) hashed: u64,
    /// Due slots left for later passes by the budgets (bytes, time,
    /// staging memory, video memory), each checked with its texture's
    /// bytes.
    pub(crate) left: u64,
    /// 1 if the pass left a texture for want of room: staging memory past
    /// [`PASS_STAGING`], or video memory past [`PASS_VIDEO_SHARE`].
    pub(crate) full: u64,
    /// Microseconds the pass took.
    pub(crate) us: u64,
}

/// Passes since the last log line: when the window began, how many, their
/// counts summed, and the longest's microseconds.
#[derive(Default)]
struct PassWindow {
    began: Option<std::time::Instant>,
    passes: u64,
    sum: PassCounts,
    longest: u64,
}

static PASS_WINDOW: Mutex<PassWindow> = Mutex::new(PassWindow {
    began: None,
    passes: 0,
    sum: PassCounts {
        looked: 0,
        uploaded: 0,
        bytes: 0,
        away: 0,
        checked: 0,
        changed: 0,
        hashed: 0,
        left: 0,
        full: 0,
        us: 0,
    },
    longest: 0,
});

/// How often the passes are summed up in the log.
const PASS_LOG: std::time::Duration = std::time::Duration::from_secs(5);

impl HostHeap {
    /// The queue's part of the title's write of slot `index` (`view`: the
    /// record written and its view, if it is one of ours).
    fn queue_texture(&self, index: u32, view: Option<(u64, &View)>) {
        if self.sampler || self.tabled.is_empty() {
            return;
        }
        let held = view
            .filter(|(_, v)| fills_texture(v))
            .map(|(key, v)| (key, matches!(v, View::Placed(..))));
        self.queue_slot(index, held, pass_frame());
    }

    /// The queue's clock at the title's frame `frame`: that frame, and one
    /// more for each pass taken in a frame that had one already (where
    /// frames do not advance, [`PASS_AGAIN`] after the last pass), so that
    /// waits counted in frames end there too.
    fn clock(&self, frame: u64) -> u64 {
        frame + self.passes_on.load(Ordering::Relaxed)
    }

    /// [`HostHeap::queue_texture`] in the title's frame `frame`, for a slot
    /// that now holds `held` (the record, and whether it is placed) or no
    /// texture the title fills. A slot tables read lately waits
    /// [`TABLED_GRACE`] frames for their draw.
    fn queue_slot(&self, index: u32, held: Option<(u64, bool)>, frame: u64) {
        let frame = self.clock(frame);
        let lately = self.tabled_within(index, frame, TABLED_FRAMES);
        let mut q = self.textures.lock().unwrap_or_else(|e| e.into_inner());
        match held {
            Some((key, placed)) => {
                let due = if lately { frame + TABLED_GRACE } else { frame };
                if q.written(index, key, placed, frame, due) && lately {
                    TABLED_WRITES[0].fetch_add(1, Ordering::Relaxed);
                }
            }
            None => q.remove(index),
        }
    }

    /// A draw read title slot `index` through a table, and brings what it
    /// views up to date (`draw.rs` `readable`): for [`TABLED_FRAMES`]
    /// frames the title's writes of the slot wait [`TABLED_GRACE`] frames
    /// for such a draw, and for [`TABLED_CHECKS`] passes leave it to them.
    /// A slot that waited for a pass is served from now on: a shader may
    /// pick it by index all the same once tables no longer read it.
    pub(crate) fn texture_slot_tabled(&self, index: u32) {
        self.tabled_in(index, pass_frame());
    }

    /// [`HostHeap::texture_slot_tabled`] in the title's frame `frame`.
    fn tabled_in(&self, index: u32, frame: u64) {
        let Some(stamp) = self.tabled.get(index as usize) else {
            return;
        };
        let now = stamp_of(self.clock(frame));
        // Once per slot and frame: tables are read by every draw.
        if stamp.load(Ordering::Relaxed) == now {
            return;
        }
        stamp.store(now, Ordering::Relaxed);
        let served = self
            .textures
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .serve(index);
        if served {
            TABLED_WRITES[1].fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Whether a draw read title slot `index` through a table within the
    /// last `frames` frames before `clock` ([`HostHeap::clock`]).
    fn tabled_within(&self, index: u32, clock: u64, frames: u64) -> bool {
        let stamp = self
            .tabled
            .get(index as usize)
            .map_or(0, |s| s.load(Ordering::Relaxed));
        stamp != 0 && u64::from(stamp_of(clock).wrapping_sub(stamp)) <= frames
    }

    /// The record ([`super::record_key`]) each of title slots `indices`
    /// was last made from: 0 for one not made, made null, or past the host
    /// heap (`super::sync_heaps`, before it mirrors records copied in as
    /// bytes; tests).
    pub(crate) fn made_keys(&self, indices: &[u32]) -> Vec<u64> {
        let made = self.made.lock().unwrap_or_else(|e| e.into_inner());
        let key = |i: u32| made.keys.get(i as usize).copied().unwrap_or(0);
        indices.iter().map(|&i| key(i)).collect()
    }

    /// What title slot `index`, which the title copied a record into as
    /// bytes (`super::sync_heaps`), holds for the shaders that may read it:
    /// `view` (`None`: no view of ours).
    pub(crate) fn copied_kind(&self, index: u32, view: Option<&View>) -> CopiedKind {
        if self.sampler {
            return CopiedKind::Sampler;
        }
        let Some(view) = view else {
            return CopiedKind::None;
        };
        if fills_texture(view) {
            let clock = self.clock(pass_frame());
            return CopiedKind::Texture {
                tabled: self.tabled_within(index, clock, TABLED_FRAMES),
            };
        }
        if buffer_range(view).is_some() {
            return CopiedKind::Buffer {
                picked: self.swept_lately(),
            };
        }
        CopiedKind::Other
    }

    /// Whether shaders that pick buffers at run time read this heap in this
    /// frame or the one before (its sweeps ran: [`HostHeap::picked_spans`]).
    fn swept_lately(&self) -> bool {
        let sweeps = self.picked.lock().unwrap_or_else(|e| e.into_inner());
        sweeps.began.is_some() && crate::frame::current().saturating_sub(sweeps.frame) <= 1
    }

    /// The work of a pass over the textures shaders pick from this heap by
    /// index, once a frame by the first list that asks (`draw.rs`
    /// `heap_textures`); `None` for the others, and in sampler heaps. A
    /// pass first releases the staging buffers no list holds any more
    /// (`texture::release_staging`): the staging limit is on those that
    /// live.
    pub(crate) fn texture_pass(&self) -> Option<TexturePass> {
        let pass = self.texture_pass_at(pass_frame(), std::time::Instant::now())?;
        crate::texture::release_staging(RELEASE_IDLE, RELEASE_LOOKS);
        Some(pass)
    }

    /// [`HostHeap::texture_pass`] at the title's frame `frame` and `now`.
    fn texture_pass_at(&self, frame: u64, now: std::time::Instant) -> Option<TexturePass> {
        if self.sampler || self.tabled.is_empty() {
            return None;
        }
        let mut q = self.textures.lock().unwrap_or_else(|e| e.into_inner());
        let again = q.pass.is_some_and(|(last, _)| last == frame);
        let since = q.pass_due(frame, now)?;
        if again {
            self.passes_on.fetch_add(1, Ordering::Relaxed);
        }
        let clock = self.clock(frame);
        let due = q.due(clock, PASS_SLOTS);
        // Counted (a walk of the waiting slots) only when the pass is
        // handed as many as it takes.
        let backlog = if due.len() == PASS_SLOTS {
            q.due_count(clock).saturating_sub(due.len()) as u64
        } else {
            0
        };
        let served = q.next_served(PASS_SLOTS, |index| {
            self.tabled_within(index, clock, TABLED_CHECKS)
        });
        let most = (4000 / since.as_millis().max(1)).clamp(16, 256) as u64;
        Some(TexturePass {
            frame: clock,
            due,
            served,
            most,
            backlog,
        })
    }

    /// Title slot `index`, as `pass` handed it out with `write`, was
    /// uploaded: served from now on.
    pub(crate) fn texture_slot_done(&self, index: u32, write: u64) {
        let mut q = self.textures.lock().unwrap_or_else(|e| e.into_inner());
        q.done(index, write);
    }

    /// Nothing to upload at title slot `index` (as `pass` handed it out
    /// with `write`) yet: looked at again later ([`TextureQueue::later`]).
    pub(crate) fn texture_slot_later(&self, index: u32, write: u64, pass: &TexturePass) {
        let mut q = self.textures.lock().unwrap_or_else(|e| e.into_inner());
        q.later(index, write, pass.frame, pass.most);
    }

    /// Title slot `index` holds no texture the title fills: out of the
    /// queue until written again. `write`: as the pass was handed it;
    /// `None` for a served slot.
    pub(crate) fn texture_slot_dropped(&self, index: u32, write: Option<u64>) {
        let mut q = self.textures.lock().unwrap_or_else(|e| e.into_inner());
        q.dropped(index, write);
    }

    /// A pass checked served slots up to `last`: the next goes on after it.
    pub(crate) fn textures_checked(&self, last: u32) {
        self.textures
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .cursor = last;
    }

    /// Counts what `pass` did; every [`PASS_LOG`], logs the passes since
    /// the last line per pass (a pass a frame), and the queue.
    pub(crate) fn texture_pass_done(&self, pass: &TexturePass, counts: PassCounts) {
        // A pass that left due slots for later passes leaves a backlog: the
        // next one may spend more ([`PASS_LOADING`]).
        // Those it was not handed count too ([`TexturePass::left`]).
        LOADING.store(loading_after(pass.left(&counts)), Ordering::Relaxed);
        let now = std::time::Instant::now();
        let mut w = PASS_WINDOW.lock().unwrap_or_else(|e| e.into_inner());
        let began = *w.began.get_or_insert(now);
        w.passes += 1;
        let s = &mut w.sum;
        s.looked += counts.looked;
        s.uploaded += counts.uploaded;
        s.bytes += counts.bytes;
        s.away += counts.away;
        s.checked += counts.checked;
        s.changed += counts.changed;
        s.hashed += counts.hashed;
        s.left += counts.left;
        s.full += counts.full;
        s.us += counts.us;
        w.longest = w.longest.max(counts.us);
        // The first pass, then a line a window.
        let first = !LOGGED.swap(true, Ordering::Relaxed);
        if !first && now.saturating_duration_since(began) < PASS_LOG {
            return;
        }
        let census = self
            .textures
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .census(pass.frame);
        let window = std::mem::take(&mut *w);
        drop(w);
        let n = window.passes.max(1) as f64;
        let per = |v: u64| (v as f64 / n * 10.0).round() / 10.0;
        let s = window.sum;
        let (budget, usage) = host()
            .and_then(|h| crate::dxgi::video_memory(&h.device))
            .unwrap_or_default();
        let (released, released_bytes) = crate::texture::released_total();
        tracing::info!(
            passes = window.passes,
            seconds = now.saturating_duration_since(began).as_secs(),
            uploaded = per(s.uploaded),
            mib = per(s.bytes >> 20),
            away = per(s.away),
            left = per(s.left),
            full_passes = s.full,
            checked = per(s.checked),
            changed = per(s.changed),
            hashed = per(s.hashed),
            ms = per(s.us) / 1000.0,
            longest_ms = window.longest / 1000,
            waiting = census[0],
            due = census[1],
            looked_before = census[2],
            served = census[3],
            oldest_frames = census[4],
            tabled_writes = TABLED_WRITES[0].load(Ordering::Relaxed),
            tabled_out = TABLED_WRITES[1].load(Ordering::Relaxed),
            staging_mib = crate::texture::staged_total() >> 20,
            staging_limit_mib = pass_staging() >> 20,
            released,
            released_mib = released_bytes >> 20,
            budget_mib = heap_textures_budget().unwrap_or(0) >> 20,
            video_mib = usage >> 20,
            video_budget_mib = budget >> 20,
            "textures shaders pick by index, per pass (a frame)"
        );
    }

    /// The queue of textures picked by index (tests).
    #[cfg(test)]
    pub(crate) fn texture_queue(&self) -> std::sync::MutexGuard<'_, TextureQueue> {
        self.textures.lock().unwrap_or_else(|e| e.into_inner())
    }
}

/// A texture a pass over the textures shaders pick by index uploads
/// (`what`: for the first time, or again after a change), at debug level
/// under its own target, `xtp_gpu::heap_textures`
/// (`XTP_LOG=info,xtp_gpu::heap_textures=debug`): its title slot, the
/// address the title placed it at (none: a texture object), its host size
/// and format. To tell whether a given texture the title fills (by its
/// address) is taken.
pub(crate) fn note_heap_texture(index: u32, handle: u64, texture: &ID3D12Resource, what: &str) {
    if !tracing::enabled!(target: "xtp_gpu::heap_textures", tracing::Level::DEBUG) {
        return;
    }
    let placed = match super::view(handle) {
        Some(View::Placed(_, _, desc)) => Some(u64::from(desc[10]) | u64::from(desc[11]) << 32),
        _ => None,
    };
    // SAFETY: a live resource.
    let d = unsafe { texture.GetDesc() };
    tracing::debug!(
        target: "xtp_gpu::heap_textures",
        slot = index,
        placed = ?placed.map(|a| format!("{a:#x}")),
        width = d.Width,
        height = d.Height,
        format = d.Format.0,
        "a texture shaders pick by index: {what}"
    );
}

/// Whether the passes have logged their first line.
static LOGGED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

#[cfg(test)]
impl TextureQueue {
    /// Slots waiting for their first upload (tests).
    pub(crate) fn waiting_slots(&self) -> Vec<u32> {
        self.waiting.keys().copied().collect()
    }

    /// Slots served (tests).
    pub(crate) fn served_slots(&self) -> Vec<u32> {
        self.served.keys().copied().collect()
    }
}

/// Makes a raw view of `window` of `buffer` at `at`: unordered access, or
/// a shader resource view.
fn raw_view(
    buffer: &ID3D12Resource,
    window: RawWindow,
    unordered: bool,
    at: D3D12_CPU_DESCRIPTOR_HANDLE,
) {
    let Some(host) = host() else { return };
    // SAFETY: a live buffer holding the window (the view just made reads
    // those bytes of it) and a slot of our heap.
    unsafe {
        if unordered {
            let desc = D3D12_UNORDERED_ACCESS_VIEW_DESC {
                Format: DXGI_FORMAT_R32_TYPELESS,
                ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
                Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                    Buffer: D3D12_BUFFER_UAV {
                        FirstElement: window.first,
                        NumElements: window.dwords,
                        Flags: D3D12_BUFFER_UAV_FLAG_RAW,
                        ..Default::default()
                    },
                },
            };
            host.device
                .CreateUnorderedAccessView(buffer, None::<&ID3D12Resource>, Some(&desc), at);
        } else {
            let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
                Format: DXGI_FORMAT_R32_TYPELESS,
                ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
                Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                    Buffer: D3D12_BUFFER_SRV {
                        FirstElement: window.first,
                        NumElements: window.dwords,
                        StructureByteStride: 0,
                        Flags: D3D12_BUFFER_SRV_FLAG_RAW,
                    },
                },
            };
            host.device
                .CreateShaderResourceView(buffer, Some(&desc), at);
        }
    }
}

/// The host resource a slot made by [`make`] views, if any.
type Kept = Option<ID3D12Resource>;

/// Counts a buffer view the host cannot make because its buffer's start in
/// the host copy (`offset`) is not on one of its elements (of `element`
/// bytes): the slot reads zeros. A console view starts at any address, and
/// the title places structured views at ring positions inside a big buffer
/// (VIctory's 86a5ed6: the meshlet culling's 456-byte constants, so every
/// meshlet was culled; watch.md item 180). Logged at the first and each
/// power of two, with the element size: how many a frame, and of what.
/// shortcut: such a view needs a copy of its own bytes (86a5ed6's
/// `shadow::for_view`), which is not made here.
fn off_element(unordered: bool, offset: u64, element: u64) {
    static COUNT: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];
    let n = COUNT[usize::from(unordered)].fetch_add(1, Ordering::Relaxed) + 1;
    if n.is_power_of_two() {
        tracing::warn!(
            count = n,
            unordered,
            offset,
            element,
            "buffer view not on an element of its host copy: null"
        );
    }
}

/// Counts a structured view `XTP_RAW_STRUCTURED` leaves structured because
/// its bytes do not start on 16 bytes of its copy (watch.md item 200: such
/// a view is still read wrong on NVIDIA where a shader reads its slot raw;
/// through the stride table its raw view starts on any dword).
fn off_16(start: u64, element: u64) {
    static COUNT: AtomicU64 = AtomicU64::new(0);
    let n = COUNT.fetch_add(1, Ordering::Relaxed) + 1;
    if n.is_power_of_two() {
        tracing::warn!(
            count = n,
            start,
            element,
            "a structured view off 16 bytes of its host copy stays structured"
        );
    }
}

/// A buffer view whose buffer does not start on one of its elements in
/// the host copy (`start`: the view's first byte there; `elements` of
/// `element` bytes), which the host cannot count a view from: its bytes go
/// to the runtime stride table, whose raw views start on any dword, and
/// with `XTP_RAW_STRUCTURED=1` the slot gets a raw view of them where they
/// start on 16 bytes. Else the slot is made null, as before, and counted.
/// `origin`: the title address of the copy's first byte.
/// The idea of five-star-recomp's 955a679 and 7bd1ca8 (watch.md item 201:
/// such a view becomes a raw view at its byte, not a null one), written
/// anew.
fn off_element_fit(
    buffer: &ID3D12Resource,
    start: u64,
    elements: u32,
    element: u64,
) -> Option<(u64, u64)> {
    // SAFETY: a live buffer.
    let width = unsafe { buffer.GetDesc() }.Width;
    let fit = width
        .saturating_sub(start)
        .checked_div(element)
        .map_or(0, |left| u64::from(elements).min(left));
    if fit == 0 || !start.is_multiple_of(4) {
        return None;
    }
    Some((fit, fit * element))
}

fn off_element_view(
    copy: &Arc<Shadow>,
    origin: u64,
    (start, elements, element): (u64, u32, u64),
    unordered: bool,
    at: D3D12_CPU_DESCRIPTOR_HANDLE,
    viewed: Option<&mut Option<Viewed>>,
) -> Option<Kept> {
    let Some((fit, len)) = off_element_fit(&copy.buffer, start, elements, element) else {
        off_element(unordered, start, element);
        return None;
    };
    let _ = fit;
    if let Some(v) = viewed {
        *v = Some(Viewed {
            copy: copy.clone(),
            title: origin + start,
            start,
            len,
            element,
            raw: false,
            unordered,
        });
    }
    let window = raw_window_of(start, len).filter(|_| xtp_shader::emit::raw_structured());
    let Some((first, dwords)) = window else {
        off_element(unordered, start, element);
        return None;
    };
    raw_view(
        &copy.buffer,
        RawWindow {
            first,
            dwords,
            bias: 0,
        },
        unordered,
        at,
    );
    Some(Some(copy.buffer.clone()))
}

/// Makes a buffer view's elements fit its host copy `copy` (`first`,
/// `elements` of `element` bytes, the first moved to the copy already):
/// none if no element does, or if the host refuses such a view (a stride
/// past D3D12's 2048, a typeless format without the raw flag), which a
/// view past its resource's end or of no elements is too. A view the host
/// refuses can remove the device; the console reads zeros past a view's
/// end, as the cut view does (VIctory #193, f99b792; the `buffer_view_ok`
/// of fxceless's VIctory fork, 578d459).
///
/// A view running past its copy is cut, not made null as VIctory's main
/// does since 4051149 (watch.md item 182): the device removal it guards
/// against comes from a view past its resource, and the cut view lies
/// wholly inside its copy; it reads the bytes the console reads, and zeros
/// past the copy (D3D12's bounds), where a null view would read zeros for
/// the valid part too, the very failure of the culling's zero reads.
///
/// With `XTP_STRIDE_TABLE` and `XTP_RUN_TIME_VIEWS` off too, so the off
/// path is not byte for byte 8d503f9's, on purpose (Codex reviews of
/// 54ab53d and 244cf78 asked). 8d503f9 handed the title's description to
/// `CreateShaderResourceView` as it was: 100 elements asked where the copy
/// holds 2 made a view of 100. D3D12 requires a buffer view to lie inside
/// its resource (`FirstElement + NumElements` elements of the element size
/// within the buffer's width), a structured view's format to be
/// `DXGI_FORMAT_UNKNOWN` and a raw view's `R32_TYPELESS`; a description
/// that breaks one is invalid: the debug layer rejects it, and without it
/// the descriptor is undefined (a read through it may fault and remove the
/// device). Every view this changes, and every format [`repair_format`]
/// changes, is such a description, so a view 8d503f9 made validly is made
/// the same; the views that differ are those whose 8d503f9 descriptor was
/// invalid. Making them so again, switches off, would bring back
/// undefined descriptors for no gain: the cut view reads what the console
/// reads, and zeros past the copy.
fn fit_view(
    copy: &ID3D12Resource,
    first: u64,
    elements: &mut u32,
    element: u64,
    (stride, format, raw): (u32, u32, bool),
) -> Option<()> {
    const MAX_ELEMENTS: u32 = 1 << D3D12_REQ_BUFFER_RESOURCE_TEXEL_COUNT_2_TO_EXP;
    if stride > 2048 {
        once!(
            stride,
            "a structured buffer view of a stride the host refuses: null"
        );
        return None;
    }
    if raw && !first.is_multiple_of(4) {
        // shortcut: the host wants raw views on 16 bytes; such a view is
        // made as before, counted (fxceless's VIctory fork's buffer_view_ok
        // refuses it).
        static OFF: AtomicU64 = AtomicU64::new(0);
        let n = OFF.fetch_add(1, Ordering::Relaxed) + 1;
        if n.is_power_of_two() {
            tracing::warn!(
                count = n,
                first,
                "a raw buffer view off 16 bytes of its host copy"
            );
        }
    }
    let typeless = crate::resource::family_members(format).is_some();
    if stride == 0 && !raw && typeless {
        once!(format, "a typeless buffer view without the raw flag: null");
        return None;
    }
    // SAFETY: a live buffer.
    let width = unsafe { copy.GetDesc() }.Width;
    let left = width.saturating_sub(first.saturating_mul(element)) / element;
    let fit = u64::from(*elements)
        .min(left)
        .min(u64::from(if stride == 0 { MAX_ELEMENTS } else { u32::MAX }));
    if fit < u64::from(*elements) {
        once!(
            elements = *elements,
            fit,
            "a buffer view past its host copy's end: cut to it"
        );
    }
    *elements = fit as u32;
    if *elements == 0 {
        once!("a buffer view of no elements in its host copy: null");
        return None;
    }
    Some(())
}

/// The host copy, title address, copy offset, and byte length viewed by a
/// buffer shader resource view `view`, or `None` for textures/samplers/null.
pub(crate) fn srv_range(view: &View) -> Option<(Arc<Shadow>, u64, u64, u64)> {
    let (address, len, desc) = match view {
        View::Shader(resource, Some(desc)) => {
            let (data, len) = crate::resource::buffer_bytes(*resource as crate::object::This)?;
            (data as u64, len, srv_desc(desc))
        }
        View::Placed(false, resource, desc) if resource[0] == 1 => {
            let len = u64::from(resource[4]) | u64::from(resource[5]) << 32;
            let address = u64::from(desc[10]) | u64::from(desc[11]) << 32;
            let words: &[u32; 10] = desc[..10].try_into().unwrap_or(&[0; 10]);
            (address, len, srv_desc(words))
        }
        _ => return None,
    };
    let shadow = crate::shadow::find(address, len)?;
    let element = element_size(&desc);
    let format = desc.Format.0 as u32;
    // SAFETY: a buffer view's union member.
    let mut buffer = unsafe { desc.Anonymous.Buffer };
    let offset = shadow.offset(address);
    let origin = address - offset;
    if element == 0 || !offset.is_multiple_of(element) {
        let start = offset.checked_add(buffer.FirstElement.checked_mul(element)?)?;
        let (_, bytes) = off_element_fit(&shadow.buffer, start, buffer.NumElements, element)?;
        return Some((shadow, origin + start, start, bytes));
    }
    buffer.FirstElement = buffer.FirstElement.checked_add(offset / element)?;
    let raw = buffer.Flags.0 & D3D12_BUFFER_SRV_FLAG_RAW.0 != 0;
    let shape = (buffer.StructureByteStride, format, raw);
    fit_view(
        &shadow.buffer,
        buffer.FirstElement,
        &mut buffer.NumElements,
        element,
        shape,
    )?;
    let start = buffer.FirstElement * element;
    let bytes = u64::from(buffer.NumElements) * element;
    Some((shadow, origin + start, start, bytes))
}

fn buffer_srv(
    address: u64,
    len: u64,
    mut desc: D3D12_SHADER_RESOURCE_VIEW_DESC,
    at: D3D12_CPU_DESCRIPTOR_HANDLE,
    viewed: Option<&mut Option<Viewed>>,
) -> Option<Kept> {
    let shadow = crate::shadow::find(address, len)?;
    let element = element_size(&desc);
    let format = desc.Format.0 as u32;
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { &mut desc.Anonymous.Buffer };
    let offset = shadow.offset(address);
    // The title address of the copy's first byte.
    let origin = address - offset;
    if element == 0 || offset % element != 0 {
        let start = offset.checked_add(buffer.FirstElement.checked_mul(element)?)?;
        let elements = buffer.NumElements;
        return off_element_view(
            &shadow,
            origin,
            (start, elements, element),
            false,
            at,
            viewed,
        );
    }
    buffer.FirstElement = buffer.FirstElement.checked_add(offset / element)?;
    let raw = buffer.Flags.0 & D3D12_BUFFER_SRV_FLAG_RAW.0 != 0;
    let shape = (buffer.StructureByteStride, format, raw);
    fit_view(
        &shadow.buffer,
        buffer.FirstElement,
        &mut buffer.NumElements,
        element,
        shape,
    )?;
    if let Some(v) = viewed {
        let start = buffer.FirstElement * element;
        *v = Some(Viewed {
            copy: shadow.clone(),
            title: origin + start,
            start,
            len: u64::from(buffer.NumElements) * element,
            element,
            raw,
            unordered: false,
        });
    }
    repair_format(&mut desc.Format, shape);
    if shape.0 > 0 && xtp_shader::emit::raw_structured() {
        // SAFETY: a buffer view's union member.
        let b = unsafe { &mut desc.Anonymous.Buffer };
        let window = raw_window_of(b.FirstElement * element, u64::from(b.NumElements) * element);
        if window.is_none() {
            off_16(b.FirstElement * element, element);
        }
        if let Some((first, dwords)) = window {
            (b.FirstElement, b.NumElements) = (first, dwords);
            b.StructureByteStride = 0;
            b.Flags = D3D12_BUFFER_SRV_FLAG_RAW;
            desc.Format = DXGI_FORMAT_R32_TYPELESS;
        }
    }
    let host = host()?;
    // SAFETY: a live buffer and a slot of our heap.
    unsafe {
        host.device
            .CreateShaderResourceView(&shadow.buffer, Some(&desc), at)
    };
    Some(Some(shadow.buffer.clone()))
}

/// With `XTP_RAW_STRUCTURED=1` ([`xtp_shader::emit::raw_structured`]), a
/// structured view's bytes (`bytes` from byte `start` of its host copy) as
/// a raw view of the same bytes: its first dword and its dwords, when they
/// start on 16 bytes (a raw view's alignment), are whole dwords and at most
/// 2^27. On NVIDIA a raw read through a structured view returns other
/// dwords or zeros (watch.md items 148, 160, 190: the lighting's
/// pre-exposure read 0), and the translator then reads every buffer by
/// element through raw handles too. A view that cannot be made raw stays
/// structured (its raw reads keep the mismatch). The stride table still
/// gives such a view its own raw view and stride.
/// Ported from fxceless's VIctory fork (commit 578d459) by fxceless; see
/// also VIctory#1762.
fn raw_window_of(start: u64, bytes: u64) -> Option<(u64, u32)> {
    if !start.is_multiple_of(16) || !bytes.is_multiple_of(4) {
        return None;
    }
    let dwords = u32::try_from(bytes / 4).ok().filter(|&d| d <= 1 << 27)?;
    Some((start / 4, dwords))
}

/// The format the host wants for a buffer view of `shape` (stride, format,
/// raw): R32 typeless for a raw view, none for a structured one; the
/// title's otherwise. The console reads the same bytes whatever format a
/// raw or structured view names; the host refuses another (the checks of
/// fxceless's VIctory fork's buffer_view_ok, 578d459).
fn repair_format(format: &mut DXGI_FORMAT, (stride, _, raw): (u32, u32, bool)) {
    if raw {
        *format = DXGI_FORMAT_R32_TYPELESS;
    } else if stride > 0 {
        *format = DXGI_FORMAT_UNKNOWN;
    }
}

/// An unordered access view of title buffer memory at `address` (`len`
/// bytes) on its host copy, as [`buffer_srv`]; false if the copy cannot be
/// written by shaders (upload memory).
/// shortcut: what shaders write stays in the host copy; title memory (and
/// so the CPU, and copies run on the CPU) does not see it.
fn buffer_uav(
    address: u64,
    len: u64,
    mut desc: D3D12_UNORDERED_ACCESS_VIEW_DESC,
    at: D3D12_CPU_DESCRIPTOR_HANDLE,
    viewed: Option<&mut Option<Viewed>>,
) -> Option<Kept> {
    let shadow = crate::shadow::find(address, len)?;
    let format = desc.Format.0 as u32;
    // SAFETY: a live buffer.
    let flags = unsafe { shadow.buffer.GetDesc() }.Flags;
    if flags.0 & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0 == 0 {
        once!("unordered access to a buffer whose host copy shaders cannot write");
        return None;
    }
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { &mut desc.Anonymous.Buffer };
    let element = if buffer.StructureByteStride > 0 {
        u64::from(buffer.StructureByteStride)
    } else if buffer.Flags.0 & D3D12_BUFFER_UAV_FLAG_RAW.0 != 0 {
        4
    } else {
        crate::resource::element(desc.Format.0 as u32).map_or(0, |(bytes, _)| bytes)
    };
    let offset = shadow.offset(address);
    // The title address of the copy's first byte.
    let origin = address - offset;
    if element == 0 || offset % element != 0 {
        let start = offset.checked_add(buffer.FirstElement.checked_mul(element)?)?;
        let elements = buffer.NumElements;
        return off_element_view(
            &shadow,
            origin,
            (start, elements, element),
            true,
            at,
            viewed,
        );
    }
    buffer.FirstElement = buffer.FirstElement.checked_add(offset / element)?;
    buffer.CounterOffsetInBytes = 0;
    let raw = buffer.Flags.0 & D3D12_BUFFER_UAV_FLAG_RAW.0 != 0;
    let shape = (buffer.StructureByteStride, format, raw);
    fit_view(
        &shadow.buffer,
        buffer.FirstElement,
        &mut buffer.NumElements,
        element,
        shape,
    )?;
    if let Some(v) = viewed {
        let start = buffer.FirstElement * element;
        *v = Some(Viewed {
            copy: shadow.clone(),
            title: origin + start,
            start,
            len: u64::from(buffer.NumElements) * element,
            element,
            raw,
            unordered: true,
        });
    }
    repair_format(&mut desc.Format, shape);
    if shape.0 > 0 && xtp_shader::emit::raw_structured() {
        // SAFETY: a buffer view's union member.
        let b = unsafe { &mut desc.Anonymous.Buffer };
        let window = raw_window_of(b.FirstElement * element, u64::from(b.NumElements) * element);
        if window.is_none() {
            off_16(b.FirstElement * element, element);
        }
        if let Some((first, dwords)) = window {
            (b.FirstElement, b.NumElements) = (first, dwords);
            b.StructureByteStride = 0;
            b.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
            desc.Format = DXGI_FORMAT_R32_TYPELESS;
        }
    }
    let host = host()?;
    // SAFETY: a live buffer and a slot of our heap.
    unsafe {
        host.device.CreateUnorderedAccessView(
            &shadow.buffer,
            None::<&ID3D12Resource>,
            Some(&desc),
            at,
        )
    };
    Some(Some(shadow.buffer.clone()))
}

/// Whether the host can make a shader (or unordered access) view of
/// `texture` in `format` (none: the texture's own). The console allows any
/// view; the host refuses (and removes the device for) views without the
/// resource's permission, of typed depth, and in a format a typed resource
/// cannot be cast to. A view of a typed texture in another format of its
/// family keeps its format where the host casts it
/// ([`crate::resource::format::view_cast`]: sRGB or linear of RGBA8, UINT
/// of UNORM, and back), and takes the texture's otherwise; each such view
/// is counted for the frame log by both formats. Before, every such view
/// took the texture's: an sRGB view of a UNORM texel of 0x80 read 0.502
/// where the console reads 0.216 (fxceless's VIctory fork, #578 and #644).
/// shortcut: a view D3D12 does not cast (FLOAT and UINT, UNORM and SNORM)
/// still reads in the texture's format; a typeless host copy would take it.
pub(crate) fn fits(
    texture: &ID3D12Resource,
    mut format: Option<&mut DXGI_FORMAT>,
    unordered: bool,
) -> bool {
    use crate::resource::format::{ViewUse, note_view_cast, view_cast};
    use crate::resource::{DEPTH_FORMATS, family};
    // SAFETY: a live resource.
    let d = unsafe { texture.GetDesc() };
    let allowed = if unordered {
        d.Flags.0 & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0 != 0
    } else {
        d.Flags.0 & D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE.0 == 0
    };
    let own = d.Format.0 as u32;
    if !allowed || DEPTH_FORMATS.iter().any(|&(typed, _)| typed == own) {
        once!(
            unordered,
            format = own,
            flags = d.Flags.0,
            "view the host does not allow"
        );
        return false;
    }
    // Every family has more than one format, its typeless one first.
    let typeless = family(own) == own && family(own + 1) == own;
    if let Some(format) = format.as_deref_mut()
        && unordered
        && let Some(&(_, linear)) = crate::resource::SRGB
            .iter()
            .find(|&&(s, _)| s == format.0 as u32)
    {
        format.0 = linear as i32;
    }
    let view = format.as_ref().map_or(0, |f| f.0 as u32);
    if view == 0 {
        if typeless {
            // (Shader views get a description: `whole_typed_view`.)
            once!(
                unordered,
                own,
                "a view without a description of a typeless host copy: null"
            );
        }
        return !typeless;
    }
    if family(view) != family(own) {
        once!(view, own, "view format of another family");
        return false;
    }
    if let Some(format) = format
        && !typeless
        && view != own
    {
        let cast = view_cast(own, view, unordered);
        let how = if unordered {
            ViewUse::Unordered
        } else {
            ViewUse::Shader
        };
        note_view_cast(own, view, how, cast);
        if !cast {
            *format = d.Format;
        }
    }
    true
}

/// Makes the host descriptor for `view` at `at`, a slot of a heap of the
/// caller's (a CPU-only one: the second handle `ClearUnorderedAccessView*`
/// takes, which a shader-visible heap cannot give): the host resource it
/// views, or `None` if the host cannot make it (or it views none).
pub(crate) fn make_view(view: &View, at: D3D12_CPU_DESCRIPTOR_HANDLE) -> Option<ID3D12Resource> {
    // shortcut: a UAV clear (`list::clear_uav`, its one caller) moves only
    // the texture of an unordered access view of a texture object to the
    // state a clear needs: a clear of a texture the title placed itself is
    // not done, as before (resolving its view with `placed_texture` there
    // would do it).
    if matches!(view, View::Placed(true, resource, _) if resource[0] != 1) {
        once!("a UAV clear of a texture the title placed itself: not done");
        return None;
    }
    make(view, at).flatten()
}

/// The host copy of title texture `r` the GPU may use now
/// ([`crate::resource::host_resource`]: none until it is resident). In
/// tests, none for the resource a test holds back as not resident yet
/// ([`NOT_RESIDENT`]): residency comes when the driver says, which a test
/// cannot wait before.
fn host_copy_of<'a>(r: crate::object::This) -> Option<&'a ID3D12Resource> {
    #[cfg(test)]
    if !r.is_null() && NOT_RESIDENT.with(std::cell::Cell::get) == r as usize {
        return None;
    }
    crate::resource::host_resource(r)
}

#[cfg(test)]
thread_local! {
    /// The resource whose host copy a test on this thread holds back as not
    /// resident yet ([`host_copy_of`]).
    static NOT_RESIDENT: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Bytes of a host constant buffer view at most: the 64 KiB a shader can
/// index (4096 vectors of 16 bytes). A larger view removes the device on
/// NVIDIA (`0x887A0001`), and title heaps hold such sizes in slots they
/// leave stale (969,472 and 1,306,368 bytes seen).
pub(crate) const CBV_BYTES: u32 = 64 << 10;

/// The bytes of the host's constant buffer view of a title one of `size`:
/// whole 256s, at most [`CBV_BYTES`].
pub(crate) fn constant_bytes(size: u32) -> u32 {
    size.min(CBV_BYTES).next_multiple_of(256)
}

/// A title constant view's `size` capped at [`CBV_BYTES`], what the host
/// reads of it. Larger sizes are logged (counted), with the largest seen.
pub(crate) fn capped_constant(address: u64, size: u32) -> u32 {
    use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
    static SEEN: AtomicU64 = AtomicU64::new(0);
    static LARGEST: AtomicU32 = AtomicU32::new(0);
    if size <= CBV_BYTES {
        return size;
    }
    let largest = LARGEST.fetch_max(size, Ordering::Relaxed).max(size);
    let count = SEEN.fetch_add(1, Ordering::Relaxed) + 1;
    if count.is_power_of_two() {
        tracing::warn!(
            count,
            address = format_args!("{address:#x}"),
            size,
            largest,
            "a constant buffer view over 64 KiB: the host views its first 64 KiB"
        );
    }
    CBV_BYTES
}

/// Makes the host descriptor for `view` at `at`: the host resource it
/// views, or `None` if it cannot.
fn make(view: &View, at: D3D12_CPU_DESCRIPTOR_HANDLE) -> Option<Kept> {
    make_with(view, at, None)
}

/// [`make`], saying in `viewed` (if asked) what a buffer view reads.
fn make_with(
    view: &View,
    at: D3D12_CPU_DESCRIPTOR_HANDLE,
    viewed: Option<&mut Option<Viewed>>,
) -> Option<Kept> {
    let host = host()?;
    let device = &host.device;
    match view {
        &View::Constant(address, size) => {
            // No address or no bytes (a null description, or a slot the
            // title cleared): a null view (`null_for`), not one of 0 bytes.
            if address == 0 || size == 0 {
                return None;
            }
            let size = capped_constant(address, size);
            let shadow = crate::shadow::find(address, u64::from(size))?;
            let gpu = shadow.gpu(address);
            if gpu % 256 != 0 {
                once!(
                    address = format_args!("{address:#x}"),
                    "constant buffer not 256-byte aligned on the host"
                );
                return None;
            }
            let desc = D3D12_CONSTANT_BUFFER_VIEW_DESC {
                BufferLocation: gpu,
                SizeInBytes: constant_bytes(size),
            };
            // SAFETY: a slot of our heap; the range lies in the host copy.
            unsafe { device.CreateConstantBufferView(Some(&desc), at) };
            Some(Some(shadow.buffer.clone()))
        }
        View::Shader(resource, desc) => {
            let resource = *resource as crate::object::This;
            if let Some(texture) = host_copy_of(resource) {
                // Without a description: the whole texture in the title's
                // format where the host copy is in another (typeless, or
                // shared with another format's texture).
                let mut desc = desc
                    .as_ref()
                    .map(srv_desc)
                    .or_else(|| crate::resource::whole_typed_view(resource));
                if !fits(texture, desc.as_mut().map(|d| &mut d.Format), false) {
                    return None;
                }
                let desc = crate::resource::view_alpha(resource, desc);
                // SAFETY: a live texture and a slot of our heap.
                unsafe {
                    device.CreateShaderResourceView(
                        texture,
                        desc.as_ref().map(std::ptr::from_ref),
                        at,
                    )
                };
                Some(Some(texture.clone()))
            } else if let (Some((data, len)), Some(desc)) =
                (crate::resource::buffer_bytes(resource), desc)
            {
                buffer_srv(data as u64, len, srv_desc(desc), at, viewed)
            } else {
                None
            }
        }
        View::Unordered(resource, _counter, desc) => {
            let resource = *resource as crate::object::This;
            let Some(texture) = host_copy_of(resource) else {
                return match (crate::resource::buffer_bytes(resource), desc) {
                    (Some((data, len)), Some(desc)) => {
                        buffer_uav(data as u64, len, uav_desc(desc), at, viewed)
                    }
                    _ => None,
                };
            };
            let mut desc = desc.as_ref().map(uav_desc);
            // A depth texture's unordered access is its colour twin's.
            let twin = crate::resource::twin_view(resource, desc);
            let (texture, mut desc) = twin.map_or((texture, desc.take()), |(t, v)| (t, Some(v)));
            if !fits(texture, desc.as_mut().map(|d| &mut d.Format), true) {
                return None;
            }
            // SAFETY: a live texture and a slot of our heap.
            unsafe {
                device.CreateUnorderedAccessView(
                    texture,
                    None::<&ID3D12Resource>,
                    desc.as_ref().map(std::ptr::from_ref),
                    at,
                )
            };
            Some(Some(texture.clone()))
        }
        View::Sampler(words) => {
            // SAFETY: 13 words in the host's layout.
            let desc = unsafe {
                std::ptr::from_ref(words)
                    .cast::<D3D12_SAMPLER_DESC>()
                    .read_unaligned()
            };
            // SAFETY: a slot of our sampler heap.
            unsafe { device.CreateSampler(&desc, at) };
            Some(None)
        }
        View::ConsoleSampler(words) => {
            let desc = crate::sampler::console(words, crate::sampler::aniso_point_mip(device));
            // SAFETY: a slot of our sampler heap.
            unsafe { device.CreateSampler(&desc, at) };
            Some(None)
        }
        View::Placed(unordered, resource, desc) => {
            if let Some(view) = placed_texture(view) {
                return make(&view, at);
            }
            const BUFFER: u32 = 1;
            if resource[0] != BUFFER {
                // No resource for its memory (yet): none could be made, or
                // its host copy is not resident; the next use asks again.
                once!(
                    dimension = resource[0],
                    unordered,
                    "a view of a texture the title placed itself without a host texture: null"
                );
                return None;
            }
            // The resource description's width, after the alignment.
            let len = u64::from(resource[4]) | u64::from(resource[5]) << 32;
            let address = u64::from(desc[10]) | u64::from(desc[11]) << 32;
            let words: &[u32; 10] = desc[..10].try_into().unwrap_or(&[0; 10]);
            if *unordered {
                buffer_uav(address, len, uav_desc(words), at, viewed)
            } else {
                buffer_srv(address, len, srv_desc(words), at, viewed)
            }
        }
    }
}

/// The title memory whose host copy [`make`] needs for `view` (address and
/// length, as it asks [`crate::shadow::find`]): constant buffers and
/// buffer views.
fn copy_range(view: &View) -> Option<(u64, u64)> {
    match view {
        &View::Constant(address, size) => Some((address, u64::from(size.min(CBV_BYTES)))),
        View::Shader(r, Some(_)) | View::Unordered(r, _, Some(_)) => {
            crate::resource::buffer_bytes(*r as crate::object::This)
                .map(|(data, len)| (data as u64, len))
        }
        View::Placed(_, resource, desc) if resource[0] == 1 => Some((
            u64::from(desc[10]) | u64::from(desc[11]) << 32,
            u64::from(resource[4]) | u64::from(resource[5]) << 32,
        )),
        _ => None,
    }
}

/// A shader or unordered access view of a texture the title placed in
/// memory itself, as a view of the resource made for that memory (on first
/// use). Unordered access views of such textures were null before, so what
/// shaders wrote through them was lost (in 49 of 103 sessions); they are
/// mirrored as shader views are, and their writes stay in the host copy
/// that later views of that memory share (`alias`).
pub(crate) fn placed_texture(view: &View) -> Option<View> {
    placed_texture_or_building(view).0
}

/// [`placed_texture`], and whether there is none only because another
/// thread is still making the resource for that memory.
pub(crate) fn placed_texture_or_building(view: &View) -> (Option<View>, bool) {
    use crate::resource::PlacedLookup;
    const BUFFER: u32 = 1;
    let &View::Placed(unordered, ref resource, ref desc) = view else {
        return (None, false);
    };
    if resource[0] == BUFFER {
        return (None, false);
    }
    let address = u64::from(desc[10]) | u64::from(desc[11]) << 32;
    let texture = match crate::resource::texture_lookup(address, resource) {
        PlacedLookup::Ready(r) => r as usize,
        PlacedLookup::Building => return (None, true),
        PlacedLookup::Absent => return (None, false),
    };
    let desc = desc[..10].try_into().ok();
    let made = if unordered {
        View::Unordered(texture, 0, desc)
    } else {
        View::Shader(texture, desc)
    };
    (Some(made), false)
}

/// A null view of the kind `view` is, for a slot whose view the host
/// cannot make: unordered access (to a buffer or a 2D texture) or a shader
/// view. Writes to it are dropped, reads give zeros.
fn null_for(view: &View, at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    let Some(host) = host() else { return };
    let is_buffer = |r: &usize| crate::resource::buffer_bytes(*r as crate::object::This).is_some();
    // Unordered views: whether a buffer, and the title's view dimension.
    let (buffer, dimension) = match view {
        View::Unordered(r, _, desc) => (
            is_buffer(r) || desc.is_some_and(|d| d[1] == 1),
            desc.map(|d| d[1]),
        ),
        View::Placed(true, resource, desc) => (resource[0] == 1, Some(desc[1])),
        // Shader views of buffers read nothing through a null buffer view:
        // a null 2D view where the shader expects a buffer is a mismatch
        // the hardware does not survive in general (GPU-based validation
        // reports it).
        View::Shader(r, desc) if is_buffer(r) || desc.is_some_and(|d| d[1] == 1) => {
            return null_buffer_view(at);
        }
        View::Placed(false, resource, _) if resource[0] == 1 => return null_buffer_view(at),
        // Textures: a null view of the title's dimension (1D, arrays, 3D,
        // cubes), not always 2D.
        View::Shader(_, Some(desc)) => return null_like(desc, at),
        View::Placed(false, _, desc) => {
            return null_like(desc[..10].try_into().unwrap_or(&[0; 10]), at);
        }
        View::Constant(..) => {
            let desc = D3D12_CONSTANT_BUFFER_VIEW_DESC::default();
            // SAFETY: a slot of our heap; a zero location makes a null view.
            return unsafe { host.device.CreateConstantBufferView(Some(&desc), at) };
        }
        _ => return null_view(at),
    };
    let desc = if buffer {
        D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: DXGI_FORMAT_R32_UINT,
            ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
            ..Default::default()
        }
    } else {
        null_uav_desc(dimension)
    };
    // SAFETY: a slot of our heap; a null resource is allowed with a desc.
    unsafe {
        host.device.CreateUnorderedAccessView(
            None::<&ID3D12Resource>,
            None::<&ID3D12Resource>,
            Some(&desc),
            at,
        )
    };
}

/// A null texture UAV of the title's view dimension (1D, 1D or 2D arrays,
/// 3D), 2D if it names none: a 2D null view where the shader writes a 3D
/// texture is a mismatch GPU-based validation reports (the world's wind
/// field, CS_CopyWindFieldBeforeUpdateWithSmooth, 2026-11-24).
fn null_uav_desc(dimension: Option<u32>) -> D3D12_UNORDERED_ACCESS_VIEW_DESC {
    let mut desc = D3D12_UNORDERED_ACCESS_VIEW_DESC {
        Format: DXGI_FORMAT_R32_UINT,
        ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2D,
        ..Default::default()
    };
    match dimension.map(|d| D3D12_UAV_DIMENSION(d as i32)) {
        Some(D3D12_UAV_DIMENSION_TEXTURE1D) => {
            desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE1D;
        }
        Some(D3D12_UAV_DIMENSION_TEXTURE1DARRAY) => {
            desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE1DARRAY;
            desc.Anonymous.Texture1DArray = D3D12_TEX1D_ARRAY_UAV {
                ArraySize: 1,
                ..Default::default()
            };
        }
        Some(D3D12_UAV_DIMENSION_TEXTURE2DARRAY) => {
            desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2DARRAY;
            desc.Anonymous.Texture2DArray = D3D12_TEX2D_ARRAY_UAV {
                ArraySize: 1,
                ..Default::default()
            };
        }
        Some(D3D12_UAV_DIMENSION_TEXTURE3D) => {
            desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE3D;
            desc.Anonymous.Texture3D = D3D12_TEX3D_UAV {
                WSize: u32::MAX,
                ..Default::default()
            };
        }
        _ => {}
    }
    desc
}

/// A buffer view that reads zeros (raw, as translated shaders mostly read
/// buffers).
fn null_buffer_view(at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    let Some(host) = host() else { return };
    let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
        Format: DXGI_FORMAT_R32_TYPELESS,
        ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
            Buffer: D3D12_BUFFER_SRV {
                NumElements: 1,
                Flags: D3D12_BUFFER_SRV_FLAG_RAW,
                ..Default::default()
            },
        },
    };
    // SAFETY: a slot of our heap; a null resource is allowed with a desc.
    unsafe {
        host.device
            .CreateShaderResourceView(None::<&ID3D12Resource>, Some(&desc), at)
    };
}

/// A raw unordered access buffer view that reads zeros and drops writes.
fn null_raw_uav(at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    let Some(host) = host() else { return };
    let desc = D3D12_UNORDERED_ACCESS_VIEW_DESC {
        Format: DXGI_FORMAT_R32_TYPELESS,
        ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
        Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
            Buffer: D3D12_BUFFER_UAV {
                NumElements: 1,
                Flags: D3D12_BUFFER_UAV_FLAG_RAW,
                ..Default::default()
            },
        },
    };
    // SAFETY: a slot of our heap; a null resource is allowed with a desc.
    unsafe {
        host.device.CreateUnorderedAccessView(
            None::<&ID3D12Resource>,
            None::<&ID3D12Resource>,
            Some(&desc),
            at,
        )
    };
}

/// A texture view that reads zeros with the dimension (and mip and array
/// counts) of the title's description `words`; 2D if it names none.
fn null_like(words: &[u32; 10], at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    const TEXTURES: std::ops::RangeInclusive<u32> = 2..=10;
    if !TEXTURES.contains(&words[1]) {
        return null_view(at);
    }
    let Some(host) = host() else { return };
    // One mip, one slice or cube, plane 0: fields a null view still checks.
    let mut w = [0u32; 10];
    w[1] = words[1];
    match words[1] {
        6 => {}
        7 => w[5] = 1,
        3 | 5 | 10 => (w[5], w[7]) = (1, 1),
        _ => w[5] = 1,
    }
    let mut desc = srv_desc(&w);
    desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    // SAFETY: a slot of our heap; a null resource is allowed with a desc.
    unsafe {
        host.device
            .CreateShaderResourceView(None::<&ID3D12Resource>, Some(&desc), at)
    };
}

/// A view that reads zeros, for slots whose view the host cannot make.
/// shortcut: slots with no view at all (kind unknown) get this null 2D
/// view; null buffer views there did not stop the world's low-address
/// page faults (2026-11-24, run w10), so they are not the cause.
fn null_view(at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    let Some(host) = host() else { return };
    let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
        Format: DXGI_FORMAT_R8G8B8A8_UNORM,
        ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
            Texture2D: D3D12_TEX2D_SRV {
                MipLevels: 1,
                ..Default::default()
            },
        },
    };
    // SAFETY: a slot of our heap; a null resource is allowed with a desc.
    unsafe {
        host.device
            .CreateShaderResourceView(None::<&ID3D12Resource>, Some(&desc), at)
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Constant views over 64 KiB are cut to 64 KiB, and those of no
    /// address or no bytes are null ones: in the title's slots (`make`) and
    /// in dynamic ones (`constant_at`). The device stays (NVIDIA removes it
    /// for a view over 64 KiB, VIctory #1640; WARP does not).
    #[test]
    fn constant_views_are_capped_at_64_kib_and_null_when_empty() {
        use crate::testgpu::TitleBuffer;
        let test =
            "descriptor::host::tests::constant_views_are_capped_at_64_kib_and_null_when_empty";
        assert_eq!(constant_bytes(480), 512);
        assert_eq!(constant_bytes(64 << 10), 64 << 10);
        assert_eq!(constant_bytes(969_472), 64 << 10);
        assert_eq!(constant_bytes(u32::MAX), 64 << 10);
        assert_eq!(capped_constant(0x1000, 1_306_368), CBV_BYTES);
        assert_eq!(capped_constant(0x1000, 4096), 4096);
        let Some(gpu) = crate::testgpu::device(test) else {
            return;
        };
        let host = gpu.host();
        // SAFETY: a plain device query.
        let alive = || unsafe { host.device.GetDeviceRemovedReason() }.is_ok();
        let heap = HostHeap::new(false, 4).expect("a host view heap");
        // A title buffer of 1 MiB: views of the title's oversized sizes lie
        // in its host copy.
        let buffer = TitleBuffer::new(1 << 20);
        let at = buffer.address();
        for size in [969_472, 896_256] {
            assert!(make(&View::Constant(at, size), heap.cpu(0)).is_some());
            assert!(alive(), "a view of {size} bytes");
        }
        // No bytes or no address: no view of the host's (the slot gets a
        // null one, `null_for`).
        assert!(make(&View::Constant(at, 0), heap.cpu(1)).is_none());
        assert!(make(&View::Constant(0, 256), heap.cpu(1)).is_none());
        assert!(make(&View::Constant(0, 0), heap.cpu(1)).is_none());
        let gpu_at = buffer.shadow().gpu(at);
        heap.constant_at(2, gpu_at, 969_472);
        heap.constant_at(3, gpu_at, 0);
        heap.constant_at(3, 0, 256);
        heap.null_constant(3);
        assert!(alive(), "views of the list's own memory");
    }

    /// A view whose buffer does not start on one of its elements in the
    /// host copy (watch.md items 180, 201) still gives the stride table its
    /// bytes: the entry names a raw view of them, though the slot itself
    /// stays null without `XTP_RAW_STRUCTURED`. Past the copy it is cut.
    #[test]
    fn views_off_their_elements_give_the_table_their_bytes() {
        use crate::testgpu::TitleBuffer;
        let test = "descriptor::host::tests::views_off_their_elements_give_the_table_their_bytes";
        let Some(_gpu) = crate::testgpu::device(test) else {
            return;
        };
        let heap = HostHeap::new_with(false, 8, true).expect("a host view heap");
        let buffer = TitleBuffer::new(1 << 16);
        let copy = buffer.shadow();
        // SAFETY: a live buffer.
        let width = unsafe { copy.buffer.GetDesc() }.Width;
        // Four elements of 12 bytes from byte 20 (not on an element).
        let origin = buffer.address() - copy.offset(buffer.address());
        let mut viewed = None;
        let made = off_element_view(
            &copy,
            origin,
            (20, 4, 12),
            false,
            heap.cpu(3),
            Some(&mut viewed),
        );
        assert!(made.is_none(), "null: no raw view starts at byte 20");
        let v = viewed.as_ref().expect("its bytes");
        assert_eq!((v.start, v.len, v.element), (20, 48, 12));
        assert_eq!(v.title, origin + 20);
        assert_eq!(heap.note_view(3, viewed.as_ref()), Noted::Named);
        let entry = heap.entry(3);
        assert_eq!((entry.stride, entry.bias, entry.count), (12, 4, 4));
        // Running past the copy: cut to the elements that fit.
        let mut viewed = None;
        off_element_view(
            &copy,
            origin,
            (width - 28, 4, 12),
            false,
            heap.cpu(4),
            Some(&mut viewed),
        );
        assert_eq!(viewed.map(|v| v.len), Some(24));
        // Off a dword: nothing a raw view could read.
        let mut viewed = None;
        off_element_view(
            &copy,
            origin,
            (22, 4, 12),
            false,
            heap.cpu(5),
            Some(&mut viewed),
        );
        assert!(viewed.is_none());
    }

    /// A structured view is made raw only where its bytes start on 16
    /// bytes, are whole dwords and fit one view.
    #[test]
    fn structured_views_go_raw_on_16_bytes() {
        assert_eq!(raw_window_of(32, 64), Some((8, 16)));
        assert_eq!(raw_window_of(0, 16 * 456), Some((0, 4 * 456)));
        assert_eq!(raw_window_of(36, 64), None, "off 16 bytes");
        assert_eq!(raw_window_of(32, 62), None, "not whole dwords");
        assert_eq!(raw_window_of(0, 4 << 27), Some((0, 1 << 27)));
        assert_eq!(raw_window_of(0, (4 << 27) + 4), None, "too many dwords");
    }

    /// A null texture UAV keeps the title's dimension (3D for the world's
    /// wind field), and is 2D when the title names none.
    #[test]
    fn null_uavs_keep_the_titles_dimension() {
        let d = null_uav_desc(Some(8));
        assert_eq!(d.ViewDimension, D3D12_UAV_DIMENSION_TEXTURE3D);
        // SAFETY: the member the dimension names.
        assert_eq!(unsafe { d.Anonymous.Texture3D.WSize }, u32::MAX);
        let d = null_uav_desc(Some(5));
        assert_eq!(d.ViewDimension, D3D12_UAV_DIMENSION_TEXTURE2DARRAY);
        // SAFETY: the member the dimension names.
        assert_eq!(unsafe { d.Anonymous.Texture2DArray.ArraySize }, 1);
        let dimension = |d| null_uav_desc(d).ViewDimension;
        assert_eq!(dimension(Some(2)), D3D12_UAV_DIMENSION_TEXTURE1D);
        assert_eq!(dimension(Some(4)), D3D12_UAV_DIMENSION_TEXTURE2D);
        assert_eq!(dimension(None), D3D12_UAV_DIMENSION_TEXTURE2D);
        // Multisampled UAVs need a cap; they stay 2D.
        assert_eq!(dimension(Some(6)), D3D12_UAV_DIMENSION_TEXTURE2D);
    }

    /// A null 3D UAV made on the real device leaves it alive.
    #[test]
    fn null_3d_uav_is_accepted_by_the_device() {
        let Some(host) = host() else {
            return; // no host GPU
        };
        let Some(heap) = HostHeap::new(false, 4) else {
            return;
        };
        let view = View::Unordered(0, 0, Some([28, 8, 0, 0, 0, 0, 0, 0, 0, 0]));
        null_for(&view, heap.cpu(0));
        // SAFETY: a plain device query.
        assert!(unsafe { host.device.GetDeviceRemovedReason() }.is_ok());
    }

    /// The queue of textures picked by index serves the slots looked at
    /// before first, then the newest write and the oldest in turns; a slot
    /// with nothing to upload is looked at again after 1, 2, 4 frames up to
    /// the limit; what the pass reports of a slot written again meanwhile
    /// is dropped; a slot written with the record it holds stays as it is,
    /// unless a placed texture's (whose host slot the write made null); the
    /// served slots are handed out in turns; and passes come once a frame,
    /// or a while after the last where frames do not advance. After the
    /// test of fxceless's VIctory fork (578d459) of the same name.
    #[test]
    fn the_texture_queue_serves_the_newest_writes_first() {
        use std::time::{Duration, Instant};
        let mut q = TextureQueue::default();
        // Slots 10..18 written in frames 1..9, slot 5 last.
        for i in 0..8u32 {
            assert!(q.written(10 + i, 100 + u64::from(i), false, u64::from(i) + 1, 0));
        }
        assert!(q.written(5, 5, false, 9, 0));
        let order = |q: &TextureQueue, frame| -> Vec<u32> {
            q.due(frame, 100).into_iter().map(|d| d.0).collect()
        };
        // newest, oldest, newest, oldest, and so on
        assert_eq!(order(&q, 9), [5, 10, 17, 11, 16, 12, 15, 13, 14]);
        assert_eq!(q.due(9, 2).len(), 2);
        assert_eq!(q.census(9), [9, 9, 0, 0, 8]);
        let write = |q: &TextureQueue, index: u32| {
            q.due(u64::MAX, 100)
                .iter()
                .find(|d| d.0 == index)
                .expect("waiting")
                .1
        };
        // Slot 17 has nothing to upload: again next frame, then in 2, 4, 4.
        let w17 = write(&q, 17);
        let mut frame = 9;
        for wait in [1, 2, 4, 4] {
            q.later(17, w17, frame, 4);
            assert!(!order(&q, frame + wait - 1).contains(&17));
            frame += wait;
            // Due again, and ahead of every slot never looked at.
            assert_eq!(order(&q, frame)[0], 17);
        }
        assert_eq!(q.census(9)[2], 1);
        q.done(17, w17);
        assert!(!order(&q, u64::MAX).contains(&17));
        assert_eq!(q.served_slots(), [17]);
        // The title writes slot 16 again while a pass works on it: what the
        // pass then says about the old write changes nothing.
        let w16 = write(&q, 16);
        assert!(q.written(16, 116, false, 13, 0));
        q.done(16, w16);
        q.later(16, w16, 13, 4);
        q.dropped(16, Some(w16));
        assert!(order(&q, 13).contains(&16));
        assert_eq!(q.served_slots(), [17]);
        // Written with the record it waits with: as it was.
        let w15 = write(&q, 15);
        assert!(!q.written(15, 105, false, 14, 0));
        assert_eq!(write(&q, 15), w15);
        // A served texture object's slot written with its record stays
        // served (a title copying the same descriptors again); a placed
        // texture's waits again.
        assert!(!q.written(17, 107, false, 15, 0));
        assert_eq!(q.served_slots(), [17]);
        assert!(q.written(17, 107, true, 15, 0));
        assert!(q.served_slots().is_empty());
        // A write due later (a table's slot) is not handed out before.
        assert!(q.written(98, 1, false, 14, 16));
        assert!(!order(&q, 15).contains(&98));
        assert!(order(&q, 16).contains(&98));
        // A slot with no texture any more: out.
        q.remove(98);
        assert!(!order(&q, u64::MAX).contains(&98));

        // Served slots in turns from where the checks stopped, a lap at
        // most, those to skip left out; a slot gone is dropped.
        let mut q = TextureQueue::default();
        for i in 1..=5u32 {
            q.written(i, u64::from(i), false, 1, 0);
            assert!(q.serve(i));
        }
        assert_eq!(q.next_served(3, |_| false), [1, 2, 3]);
        q.cursor = 3;
        assert_eq!(q.next_served(10, |i| i == 5), [4, 1, 2, 3]);
        q.dropped(4, None);
        assert_eq!(q.next_served(10, |_| false), [5, 1, 2, 3]);
        assert_eq!(q.census(1), [0, 0, 0, 4, 0]);

        // Once a frame, or a while on where frames do not advance.
        let t = Instant::now();
        let ms = Duration::from_millis;
        assert_eq!(q.pass_due(1, t), Some(Duration::MAX));
        assert_eq!(q.pass_due(1, t + ms(50)), None);
        assert_eq!(q.pass_due(2, t + ms(60)), Some(ms(60)));
        assert_eq!(q.pass_due(2, t + ms(60) + PASS_AGAIN), Some(PASS_AGAIN));
    }

    /// A slot a draw read through a table stops waiting for a pass and is
    /// served, though passes leave it to such draws while they read it; the
    /// title's next write of it waits for such a draw, and is due after the
    /// grace frames if none reads it; long after the last table, a write is
    /// due at once. After the test of fxceless's VIctory fork (578d459) of
    /// the same name.
    #[test]
    fn slots_read_through_tables_leave_the_texture_queue() {
        use std::time::Instant;
        let Some(heap) = HostHeap::new(false, 8) else {
            return; // no host GPU
        };
        let frame = 100;
        let due = |at: u64| -> Vec<u32> {
            heap.texture_queue()
                .due(at, 8)
                .into_iter()
                .map(|d| d.0)
                .collect()
        };
        heap.queue_slot(2, Some((2, false)), frame);
        heap.queue_slot(3, Some((3, false)), frame);
        assert_eq!(due(frame), [3, 2]);
        heap.tabled_in(3, frame);
        assert_eq!(due(frame), [2]);
        assert_eq!(heap.texture_queue().served_slots(), [3]);
        // Read through a table lately: no pass checks it.
        let pass = heap.texture_pass_at(frame, Instant::now()).expect("a pass");
        assert!(pass.served.is_empty());
        assert_eq!(pass.due.iter().map(|d| d.0).collect::<Vec<_>>(), [2]);
        assert!(
            heap.texture_pass_at(frame, Instant::now()).is_none(),
            "one a frame"
        );
        // Written again in the next frame: due once the grace frames have
        // passed without a draw that reads it.
        heap.queue_slot(3, Some((30, false)), frame + 1);
        assert_eq!(due(frame + TABLED_GRACE), [2]);
        assert_eq!(due(frame + 1 + TABLED_GRACE), [3, 2]);
        // The draw comes after all: served again.
        heap.tabled_in(3, frame + 2);
        assert_eq!(due(u64::MAX), [2]);
        // No texture there any more: out.
        heap.queue_slot(2, None, frame + 2);
        assert!(due(u64::MAX).is_empty());
        // Long after the last table, a write is due at once, and once
        // served it is checked.
        let later = frame + 3 + TABLED_FRAMES;
        heap.queue_slot(3, Some((31, false)), later);
        assert_eq!(due(later), [3]);
        assert!(heap.texture_queue().serve(3));
        let pass = heap.texture_pass_at(later, Instant::now()).expect("a pass");
        assert_eq!(pass.served, [3]);
    }

    /// `XTP_HEAP_TEXTURES`: MiB a pass stages, 64 by default; 0, `off` or
    /// `false`, no pass.
    #[test]
    fn heap_texture_budgets_read_as_mib() {
        assert_eq!(heap_textures_budget_of(None), Some(64 << 20));
        assert_eq!(heap_textures_budget_of(Some(" 16 ")), Some(16 << 20));
        assert_eq!(heap_textures_budget_of(Some("much")), Some(64 << 20));
        for off in ["0", "off", "False"] {
            assert_eq!(heap_textures_budget_of(Some(off)), None, "{off}");
        }
    }

    /// Passes make host textures within 85 % of the local video memory
    /// budget (heaptex-A: 13.3 GB resident on a 16 GB card shared with the
    /// desktop, 1-11 s a frame on the GPU), each with its bytes: a 64 MiB
    /// texture does not pass with 1 MiB left. Unknown, any. Staging
    /// buffers within the staging limit and what is left of the pass's
    /// bytes, but for the first of a pass, which may take more than they.
    #[test]
    fn passes_make_textures_and_staging_buffers_within_their_room() {
        const GB: u64 = 1 << 30;
        const MIB: u64 = 1 << 20;
        assert_eq!(video_room(None), u64::MAX);
        assert_eq!(
            video_room(Some((14 * GB, 10 * GB))),
            14 * GB / 100 * 85 - 10 * GB,
            "85 %: 11.9 GB"
        );
        assert_eq!(video_room(Some((14 * GB, 12 * GB))), 0);
        assert_eq!(video_room(Some((0, 0))), 0, "no budget at all");
        let left = video_room(Some((100 * MIB, 84 * MIB)));
        assert_eq!(left, MIB);
        assert!(64 * MIB > left, "a 64 MiB texture waits");
        let room = |staged, spent, budget| staging_room_within(PASS_STAGING, staged, spent, budget);
        assert_eq!(room(0, 0, 64 * MIB), PASS_STAGING, "the first");
        assert_eq!(room(0, 60 * MIB, 64 * MIB), 4 * MIB);
        assert_eq!(room(0, 64 * MIB, 64 * MIB), 0);
        assert_eq!(room(PASS_STAGING - MIB, 0, 64 * MIB), MIB);
        assert_eq!(room(PASS_STAGING + MIB, 0, 64 * MIB), 0);
        // A raised limit (XTP_PASS_STAGING) leaves room past the default.
        assert_eq!(
            staging_room_within(4 * GB, PASS_STAGING + MIB, 0, 64 * MIB),
            3 * GB - MIB
        );
    }

    /// A pass that leaves due slots for later passes leaves a backlog: the
    /// next may spend `PASS_LOADING` times as much, and the passes go back
    /// to the base once one leaves none. One whose next texture did not
    /// fit in what was left of the base (six 10 MiB staging buffers of
    /// 64 MiB, the seventh refused) leaves some: before, the rule was the
    /// base spent, which its 60 MiB were not (Codex's review of c44a716, 4).
    #[test]
    fn passes_spend_more_while_a_backlog_waits() {
        const MIB: u64 = 1 << 20;
        let base = 64 * MIB;
        assert_eq!(pass_budget(base, false), base);
        assert_eq!(pass_budget(base, true), PASS_LOADING * base);
        // Eight 10 MiB textures due, staged as a pass does.
        let (mut spent, mut left) = (0, 0);
        for _ in 0..8 {
            if 10 * MIB <= staging_room_within(PASS_STAGING, spent, spent, base) {
                spent += 10 * MIB;
            } else {
                left += 1;
            }
        }
        assert_eq!((spent, left), (60 * MIB, 2));
        assert!(loading_after(left), "slots left: more");
        assert!(!loading_after(0), "none left: back to the base");
    }

    /// Slots due past those a pass is handed ([`PASS_SLOTS`]) are a backlog
    /// all the same: a pass that did all it was handed leaves them, and the
    /// next spends more. Before, only the slots it was handed and left
    /// counted, and the next went back to the base with slots due (Codex's
    /// review of db37732, 4).
    #[test]
    fn texture_slots_due_past_those_a_pass_is_handed_are_a_backlog() {
        use std::time::Instant;
        let test =
            "descriptor::host::tests::texture_slots_due_past_those_a_pass_is_handed_are_a_backlog";
        let Some(_gpu) = crate::testgpu::device(test) else {
            return;
        };
        let frame = 300;
        let n = PASS_SLOTS as u32 + 2;
        let heap = HostHeap::new(false, n + 1).expect("a host heap");
        for index in 1..=n {
            heap.queue_slot(index, Some((u64::from(index), false)), frame);
        }
        let pass = heap.texture_pass_at(frame, Instant::now()).expect("a pass");
        assert_eq!((pass.due.len(), pass.backlog), (PASS_SLOTS, 2));
        // All it was handed uploaded: two left all the same.
        let all = PassCounts {
            looked: PASS_SLOTS as u64,
            uploaded: PASS_SLOTS as u64,
            ..PassCounts::default()
        };
        assert_eq!(pass.left(&all), 2);
        assert!(loading_after(pass.left(&all)));
        // Fewer due than a pass takes: no backlog.
        let few = HostHeap::new(false, 8).expect("a host heap");
        few.queue_slot(1, Some((1, false)), frame);
        let pass = few.texture_pass_at(frame, Instant::now()).expect("a pass");
        assert_eq!((pass.due.len(), pass.backlog), (1, 0));
        let one = PassCounts {
            looked: 1,
            uploaded: 1,
            ..PassCounts::default()
        };
        assert!(!loading_after(pass.left(&one)));
    }

    /// `XTP_PASS_STAGING`: MiB of staging limit, at most 4 GiB; anything
    /// else, the default.
    #[test]
    fn staging_limits_read_as_mib_up_to_four_gib() {
        const MIB: u64 = 1 << 20;
        assert_eq!(pass_staging_of(None), PASS_STAGING);
        assert_eq!(pass_staging_of(Some(" 2048 ")), 2048 * MIB);
        assert_eq!(pass_staging_of(Some("4096")), 4096 * MIB);
        assert_eq!(pass_staging_of(Some("65536")), PASS_STAGING_MOST, "capped");
        // From 2^44 MiB on: bounded before the shift, not wrapped to 0
        // (Codex's review of c44a716, 5).
        assert_eq!(pass_staging_of(Some("17592186044416")), PASS_STAGING_MOST);
        assert_eq!(
            pass_staging_of(Some(&u64::MAX.to_string())),
            PASS_STAGING_MOST
        );
        for other in ["0", "much", ""] {
            assert_eq!(pass_staging_of(Some(other)), PASS_STAGING, "{other:?}");
        }
    }

    /// Where frames do not advance (a title that does not wait for them), a
    /// pass taken a while after the last counts a frame more on the queue's
    /// clock: a slot with nothing to upload is looked at again, and a
    /// table's slot written is due once its grace frames have passed, though
    /// the title's frame stays the same. Before, both waited for good.
    #[test]
    fn waits_end_where_frames_do_not_advance() {
        use std::time::Instant;
        let Some(heap) = HostHeap::new(false, 8) else {
            return; // no host GPU
        };
        let frame = 7;
        let due = |pass: &TexturePass| pass.due.iter().map(|d| d.0).collect::<Vec<_>>();
        let mut at = Instant::now();
        heap.queue_slot(1, Some((1, false)), frame);
        let pass = heap.texture_pass_at(frame, at).expect("a pass");
        assert_eq!(due(&pass), [1]);
        heap.texture_slot_later(1, pass.due[0].1, &pass);
        assert!(heap.texture_pass_at(frame, at).is_none(), "one a frame");
        at += PASS_AGAIN;
        let pass = heap.texture_pass_at(frame, at).expect("a pass");
        assert_eq!(due(&pass), [1], "looked at again");
        // A slot a draw reads through a table, written: due after its grace
        // frames, counted in passes.
        heap.tabled_in(2, frame);
        heap.queue_slot(2, Some((2, false)), frame);
        for n in 1..=TABLED_GRACE {
            at += PASS_AGAIN;
            let pass = heap.texture_pass_at(frame, at).expect("a pass");
            assert_eq!(due(&pass).contains(&2), n == TABLED_GRACE, "pass {n}");
        }
    }

    /// A served slot a draw read through a table is left to such draws for
    /// that frame and the next only: once the title stops reading it through
    /// tables (to pick it by index), passes check it from the frame after,
    /// not 240 frames later with the image it had.
    #[test]
    fn served_slots_tables_stopped_reading_are_checked() {
        use std::time::Instant;
        let Some(heap) = HostHeap::new(false, 8) else {
            return; // no host GPU
        };
        let frame = 50;
        heap.queue_slot(4, Some((4, false)), frame);
        heap.tabled_in(4, frame);
        assert_eq!(heap.texture_queue().served_slots(), [4]);
        let served = |f: u64| {
            heap.texture_pass_at(f, Instant::now())
                .expect("a pass")
                .served
        };
        assert!(served(frame).is_empty(), "read through a table this frame");
        assert!(served(frame + 1).is_empty(), "and the frame before");
        assert_eq!(served(frame + 2), [4], "then the passes check it");
    }

    /// A slot the title wrote a texture object's view into before its host
    /// copy was resident is null, and made over that copy once it is: the
    /// record it holds is the same, only the host copy tells. Before, it
    /// stayed null for good (the record was what it was made from).
    #[test]
    fn texture_object_slots_are_made_once_their_copy_is_resident() {
        use crate::resource::tests_support::{committed, sampled};
        let test =
            "descriptor::host::tests::texture_object_slots_are_made_once_their_copy_is_resident";
        let Some(_gpu) = crate::testgpu::device(test) else {
            return;
        };
        let object = committed(&sampled(16, 16, 28));
        let started = std::time::Instant::now();
        while crate::resource::host_resource(object).is_none() {
            assert!(started.elapsed().as_secs() < 5, "never resident");
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        let copy = crate::resource::host_resource(object).map(|t| t.as_raw() as usize);
        // Not resident yet when the title writes the view.
        NOT_RESIDENT.with(|held| held.set(object as usize));
        let heap = crate::descriptor::TestHeap::views(4);
        let title = crate::descriptor::heap(heap.0);
        let host = title.host().expect("a host heap");
        let handle = title.start + 2 * 32;
        // Format, 2D, the default mapping, then (from word 4) the first mip
        // and the count.
        let texture_2d = [28, 4, 0x1688, 0, 0, 1, 0, 0, 0, 0];
        crate::descriptor::create_srv(std::ptr::null(), object, &texture_2d, handle);
        let kept = |h: &HostHeap| {
            h.kept.lock().unwrap_or_else(|e| e.into_inner())[2]
                .as_ref()
                .map(|t| t.as_raw() as usize)
        };
        assert_eq!(kept(host), None, "null while not resident");
        assert!(matches!(
            host.mirror(2, handle, true),
            Some(View::Shader(..))
        ));
        assert_eq!(kept(host), None, "still not resident");
        NOT_RESIDENT.with(|held| held.set(0));
        host.mirror(2, handle, true);
        assert_eq!(kept(host), copy, "made over its host copy");
        host.mirror(2, handle, true);
        assert_eq!(kept(host), copy);
        drop(heap);
        type Release = extern "system" fn(crate::object::This) -> u32;
        crate::object::slot::<Release>(object, 0x10)(object);
    }

    /// Dynamic slots stay taken until given back, and runs do not wrap.
    #[test]
    fn dynamic_slots_are_held_until_released() {
        let Some(heap) = HostHeap::new(false, 4) else {
            return; // no host GPU
        };
        let a = heap.dynamic(heap.dynamic - 100).unwrap();
        let b = heap.dynamic(60).unwrap();
        assert_eq!((a, b), (4, 4 + heap.dynamic - 100));
        // 40 left at the end: 50 do not fit until something is given back.
        assert_eq!(heap.dynamic(50), None);
        heap.release(a, 50);
        assert_eq!(heap.dynamic(50), Some(4));
        assert_eq!(heap.dynamic(40), Some(4 + heap.dynamic - 40));
    }

    #[test]
    fn dynamic_runs_merge_across_the_last_allocation_but_not_the_heap_boundary() {
        let mut s = Slots::new(16);
        assert_eq!(s.reserve(6), Some(0));
        assert_eq!(s.reserve(3), Some(6));
        s.release(4, 9);
        assert_eq!(s.reserve(12), Some(4));
        s.release(0, 2);
        s.release(14, 16);
        assert_eq!(s.reserve(3), None);
        assert_eq!(s.used, 12);
    }

    #[test]
    fn dynamic_tables_reuse_small_holes_before_splitting_large_ones() {
        let mut s = Slots::new(512);
        assert_eq!(s.reserve(512), Some(0));
        s.release(0, 258);
        s.release(300, 480);
        assert_eq!(s.reserve(179), Some(300));
        assert_eq!(s.reserve(258), Some(0));
        assert_eq!(s.reserve(1), Some(479));
        assert_eq!(s.used, 512);
    }

    /// A capped view heap keeps 2^18 of its slots dynamic and the rest for
    /// the title; small limits keep a title slot and never underflow
    /// (VIctory #1349).
    #[test]
    fn capped_view_heaps_split_their_slots() {
        assert_eq!(capped_split(PORTABLE_VIEWS), (737_856, 1 << 18));
        assert_eq!(capped_split(1024), (1, 1023));
        assert_eq!(capped_split(1), (1, 0));
        assert_eq!(capped_split(0), (0, 0));
    }

    /// A view heap of BTP 6's size, past the 1,000,000 slots many hosts
    /// cap at: capped, it has 2^18 dynamic slots, all free at first; a host
    /// that takes it whole keeps every title slot.
    #[test]
    fn a_heap_of_the_titles_size_keeps_its_dynamic_slots() {
        const TITLE: u32 = 1_245_184;
        let Some(heap) = HostHeap::new(false, TITLE) else {
            return; // no host GPU
        };
        if heap.count() < TITLE {
            assert_eq!((heap.count(), heap.dynamic), capped_split(PORTABLE_VIEWS));
        } else {
            assert!(heap.dynamic >= MIN_DYNAMIC);
        }
        let first = heap.dynamic(heap.dynamic).expect("every dynamic slot free");
        assert_eq!(first, heap.count());
        heap.release(first, heap.dynamic);
    }

    /// The peak of dynamic slots in use is kept, and ranges go back in one
    /// batch.
    #[test]
    fn dynamic_slots_count_their_peak_and_go_back_in_a_batch() {
        let Some(heap) = HostHeap::new(false, 4) else {
            return; // no host GPU
        };
        let a = heap.dynamic(10).unwrap();
        let b = heap.dynamic(1).unwrap();
        let c = heap.dynamic(5).unwrap();
        assert_eq!(heap.dynamic_use(), (16, 16));
        heap.release_all(&[(a, 10), (c, 5)]);
        assert_eq!(heap.dynamic_use(), (1, 16));
        // Given back twice, or past the end: counted once.
        heap.release_all(&[(b, 1), (b, 1), (4 + heap.dynamic - 1, 5)]);
        assert_eq!(heap.dynamic_use(), (0, 16));
        assert_eq!(
            heap.dynamic(16),
            Some(a),
            "the merged range starts at the first freed slot"
        );
    }

    /// A shared raw view is made once per range and host copy; views no
    /// list holds stay for later lists and give their slots back when a
    /// table needs room.
    #[test]
    fn shared_views_are_made_once_and_make_room_when_slots_run_out() {
        use crate::testgpu::TitleBuffer;
        let test =
            "descriptor::host::tests::shared_views_are_made_once_and_make_room_when_slots_run_out";
        let Some(_gpu) = crate::testgpu::device(test) else {
            return;
        };
        let heap = HostHeap::new(false, 4).expect("a host view heap");
        let buffer = TitleBuffer::new(1 << 16);
        let copy = buffer.shadow();
        let a = heap.shared_raw(&copy, 0, 64).unwrap();
        assert_eq!(heap.shared_raw(&copy, 0, 64), Some(a), "the same range");
        let b = heap.shared_raw(&copy, 16, 48).unwrap();
        assert_ne!(a, b);
        assert_eq!(heap.shared_count(), (2, 0));
        // One chunk taken for both.
        assert_eq!(heap.dynamic_use().0, SHARED_CHUNK);
        heap.give_back(&[a, b]);
        assert_eq!(heap.shared_count(), (2, 1), "a is held once more");
        heap.give_back(&[a]);
        assert_eq!(heap.shared_count(), (2, 2));
        assert_eq!(
            heap.shared_raw(&copy, 0, 64),
            Some(a),
            "kept for later lists"
        );
        heap.give_back(&[a]);
        // Every other slot taken: the next table takes the idle views' chunk.
        let rest = heap.dynamic(heap.dynamic - SHARED_CHUNK).unwrap();
        assert_eq!(heap.dynamic(SHARED_CHUNK), Some(4));
        assert_eq!(heap.shared_count(), (0, 0));
        assert_eq!(heap.dynamic_use().0, heap.dynamic);
        heap.release_all(&[(rest, heap.dynamic - SHARED_CHUNK), (4, SHARED_CHUNK)]);
        // A view a list holds is never taken back.
        let held = heap.shared_raw(&copy, 0, 64).unwrap();
        while heap.take(SHARED_CHUNK).is_some() {}
        assert_eq!(heap.dynamic(SHARED_CHUNK), None);
        assert_eq!(heap.shared_count(), (1, 0));
        heap.give_back(&[held]);
    }

    /// A buffer view's slot is made, for a draw, over its buffer's host
    /// copy, made first (outside the slot lock) when it is not yet; not
    /// for a mere view creation.
    #[test]
    fn buffer_slots_get_their_host_copy_made_first() {
        use crate::testgpu::TitleBuffer;
        let test = "descriptor::host::tests::buffer_slots_get_their_host_copy_made_first";
        let Some(_gpu) = crate::testgpu::device(test) else {
            return;
        };
        let heap = crate::descriptor::TestHeap::views(4);
        let host = crate::descriptor::heap(heap.0).host().expect("a host heap");
        let buffer = TitleBuffer::new(1 << 16);
        let address = buffer.address();
        assert!(
            crate::shadow::existing(address, 256).is_none(),
            "no copy yet"
        );
        let handle = crate::descriptor::heap(heap.0).start + 2 * 32;
        let words = [address as u32, (address >> 32) as u32, 256, 0];
        crate::descriptor::create_cbv(std::ptr::null(), &words, handle);
        assert!(
            crate::shadow::existing(address, 256).is_none(),
            "views wait for a draw"
        );
        assert_eq!(
            host.mirror(2, handle, true),
            Some(View::Constant(address, 256))
        );
        let copy = crate::shadow::existing(address, 256).expect("made for the draw");
        let kept = host.kept.lock().unwrap_or_else(|e| e.into_inner())[2]
            .as_ref()
            .map(|b| b.as_raw() as usize);
        assert_eq!(kept, Some(copy.buffer.as_raw() as usize));
    }

    /// Waits until title texture `r`'s host copy is resident, at most 5 s.
    /// Host copies are made resident in the background (`resource/
    /// residency.rs`: a fence the driver signals when it is done), and until
    /// then lookups find none and a slot made over the texture is null.
    /// Issue #54: the two placed-texture tests below made their slot at
    /// once, and now and then (a serial run under the debug layer on WARP,
    /// CI's gpu-debug-layer run of PR #26) before the fence: the slot was
    /// null, or `host_copy` found none.
    fn resident(r: crate::object::This) {
        let started = std::time::Instant::now();
        while crate::resource::host_resource(r).is_none() {
            assert!(started.elapsed().as_secs() < 5, "never resident");
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
    }

    /// A slot viewing a texture the title placed itself follows the
    /// resource made for that memory: when a texture the GPU writes comes
    /// there (#643), the slot is made again over its copy, although the
    /// title's record has not changed.
    #[test]
    fn placed_texture_slots_follow_their_replacement() {
        use crate::object::{S_OK, This, guid, slot};
        use crate::resource::tests_support::target;
        let test = "descriptor::host::tests::placed_texture_slots_follow_their_replacement";
        let Some(_gpu) = crate::testgpu::device(test) else {
            return;
        };
        const KIB_64: u64 = 1 << 16;
        // Freed only once nothing is placed in it any more (a failing
        // assertion leaks it rather than leave textures over freed memory).
        let memory =
            std::mem::ManuallyDrop::new(crate::resource::Memory::zeroed(KIB_64, KIB_64).unwrap());
        let address = memory.address();
        // A 16x16 RGBA8 texture placed there, and a shader view of it.
        let drawn = target(16, 16, 28);
        // SAFETY: a description is 56 bytes, 14 words.
        let mut resource: [u32; 14] = unsafe { std::mem::transmute_copy(&drawn) };
        resource[12] = 0; // not a render target: the title fills it
        let mut view = [0u32; 16];
        // Format, 2D, the default mapping, then (from word 4, 8-byte
        // aligned) the first mip and the mip count.
        view[..6].copy_from_slice(&[28, 4, 0x1688, 0, 0, 1]);
        (view[10], view[11]) = (address as u32, (address >> 32) as u32);
        let heap = crate::descriptor::TestHeap::views(4);
        let host = crate::descriptor::heap(heap.0).host().expect("a host heap");
        let handle = crate::descriptor::heap(heap.0).start + 32;
        crate::descriptor::create_placed_srv(std::ptr::null(), &resource, &view, handle);
        let kept = |h: &HostHeap| {
            h.kept.lock().unwrap_or_else(|e| e.into_inner())[1]
                .as_ref()
                .map(|t| t.as_raw() as usize)
        };
        let host_of = |r: This| crate::resource::host_resource(r).map(|t| t.as_raw() as usize);
        // The texture made for the memory, resident before the slot is made
        // (see `resident`).
        let before = crate::resource::texture_at(address, &resource).unwrap();
        resident(before);
        assert!(matches!(
            host.mirror(1, handle, true),
            Some(View::Shader(..))
        ));
        assert_eq!(
            crate::resource::texture_at(address, &resource),
            Some(before)
        );
        // Taken now: once replaced, `before` may be released at any time
        // (and its address reused); its copy stays while this holds it.
        let old = crate::resource::host_copy(before).expect("a host copy");
        let old_texture = Some(old.texture.as_raw() as usize);
        assert_eq!(kept(host), old_texture);
        // A render target comes to the same memory: the view's texture is
        // replaced by one over the target's copy.
        const RESOURCE: crate::object::Iid = guid(
            0x696442be,
            0xa72e,
            0x4059,
            [0xbc, 0x79, 0x5b, 0x5c, 0x98, 0x04, 0x0f, 0xad],
        );
        let mut rt = std::ptr::null();
        let made = crate::resource::create_at_address(
            std::ptr::null(),
            address,
            &drawn,
            0,
            std::ptr::null(),
            &RESOURCE,
            &mut rt,
        );
        assert_eq!(made, S_OK);
        resident(rt);
        assert!(matches!(
            host.mirror(1, handle, true),
            Some(View::Shader(..))
        ));
        let after = crate::resource::texture_at(address, &resource).unwrap();
        assert_eq!(
            host_of(after),
            host_of(rt),
            "texture_at took the target's copy"
        );
        assert_ne!(host_of(after), old_texture);
        assert_eq!(kept(host), host_of(after), "the slot views the replacement");
        // Nothing more changes: the slot stays as it is.
        host.mirror(1, handle, true);
        assert_eq!(kept(host), host_of(after));
        assert_eq!(slot::<extern "system" fn(This) -> u32>(rt, 0x10)(rt), 0);
        drop(heap);
        crate::resource::memory_remapped(address as usize, KIB_64 as usize);
        drop(std::mem::ManuallyDrop::into_inner(memory));
    }

    /// A slot made over a placed view's texture keeps it when another
    /// mirror finds no texture there because a thread is still making one:
    /// it does not write a null view over the good one.
    #[test]
    fn a_placed_slot_keeps_its_view_while_another_is_made() {
        use crate::resource::tests_support::{target, while_placed_building};
        let test = "descriptor::host::tests::a_placed_slot_keeps_its_view_while_another_is_made";
        let Some(_gpu) = crate::testgpu::device(test) else {
            return;
        };
        const KIB_64: u64 = 1 << 16;
        let memory =
            std::mem::ManuallyDrop::new(crate::resource::Memory::zeroed(KIB_64, KIB_64).unwrap());
        let address = memory.address();
        let drawn = target(16, 16, 28);
        // SAFETY: a description is 56 bytes, 14 words.
        let mut resource: [u32; 14] = unsafe { std::mem::transmute_copy(&drawn) };
        resource[12] = 0;
        let mut view = [0u32; 16];
        view[..6].copy_from_slice(&[28, 4, 0x1688, 0, 0, 1]);
        (view[10], view[11]) = (address as u32, (address >> 32) as u32);
        let heap = crate::descriptor::TestHeap::views(4);
        let host = crate::descriptor::heap(heap.0).host().expect("a host heap");
        let handle = crate::descriptor::heap(heap.0).start + 32;
        crate::descriptor::create_placed_srv(std::ptr::null(), &resource, &view, handle);
        let kept = |h: &HostHeap| {
            h.kept.lock().unwrap_or_else(|e| e.into_inner())[1]
                .as_ref()
                .map(|t| t.as_raw() as usize)
        };
        assert!(matches!(
            host.mirror(1, handle, true),
            Some(View::Shader(..))
        ));
        let made = kept(host);
        assert!(made.is_some(), "the slot views the placed texture");
        while_placed_building(address, &resource, || {
            host.mirror(1, handle, true);
        });
        assert_eq!(kept(host), made, "the slot kept its view");
        drop(heap);
        crate::resource::memory_remapped(address as usize, KIB_64 as usize);
        drop(std::mem::ManuallyDrop::into_inner(memory));
    }

    /// A slot made over a placed view's texture follows its memory when the
    /// memory is unmapped: the next mirror finds no texture there, and no
    /// thread making one, so the slot is made null.
    #[test]
    fn a_placed_slot_follows_its_memory_when_unmapped() {
        use crate::resource::tests_support::target;
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_DECOMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc,
            VirtualFree,
        };
        let test = "descriptor::host::tests::a_placed_slot_follows_its_memory_when_unmapped";
        let Some(_gpu) = crate::testgpu::device(test) else {
            return;
        };
        const KIB_64: usize = 1 << 16;
        // SAFETY: reserve and commit an OS-selected 64 KiB range.
        let base = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                KIB_64,
                MEM_RESERVE | MEM_COMMIT,
                PAGE_READWRITE,
            )
        };
        assert!(!base.is_null());
        let address = base as u64;
        let drawn = target(16, 16, 28);
        // SAFETY: a description is 56 bytes, 14 words.
        let mut resource: [u32; 14] = unsafe { std::mem::transmute_copy(&drawn) };
        resource[12] = 0;
        let mut view = [0u32; 16];
        view[..6].copy_from_slice(&[28, 4, 0x1688, 0, 0, 1]);
        (view[10], view[11]) = (address as u32, (address >> 32) as u32);
        let heap = crate::descriptor::TestHeap::views(4);
        let host = crate::descriptor::heap(heap.0).host().expect("a host heap");
        let handle = crate::descriptor::heap(heap.0).start + 32;
        crate::descriptor::create_placed_srv(std::ptr::null(), &resource, &view, handle);
        let kept = |h: &HostHeap| {
            h.kept.lock().unwrap_or_else(|e| e.into_inner())[1]
                .as_ref()
                .map(|t| t.as_raw() as usize)
        };
        assert!(matches!(
            host.mirror(1, handle, true),
            Some(View::Shader(..))
        ));
        assert!(kept(host).is_some(), "the slot views the placed texture");
        // SAFETY: the range was committed above and nothing else uses it.
        assert_ne!(unsafe { VirtualFree(base, KIB_64, MEM_DECOMMIT) }, 0);
        crate::resource::memory_remapped(address as usize, KIB_64);
        host.mirror(1, handle, true);
        assert_eq!(kept(host), None, "the slot no longer shows unmapped memory");
        drop(heap);
        // SAFETY: release the reservation once no view is placed in it.
        assert_ne!(unsafe { VirtualFree(base, 0, MEM_RELEASE) }, 0);
    }

    /// An unordered access view of a texture the title placed itself is
    /// made, for a draw, over the resource made for that memory, as shader
    /// views are (it was null: its writes were lost); a shader view placed
    /// at the same memory later reads the same host copy, what the GPU
    /// wrote through the first.
    #[test]
    fn placed_unordered_texture_views_are_made_over_their_memory() {
        use crate::object::This;
        use crate::resource::tests_support::target;
        let test =
            "descriptor::host::tests::placed_unordered_texture_views_are_made_over_their_memory";
        let Some(_gpu) = crate::testgpu::device(test) else {
            return;
        };
        const KIB_64: u64 = 1 << 16;
        // Freed only once nothing is placed in it any more.
        let memory =
            std::mem::ManuallyDrop::new(crate::resource::Memory::zeroed(KIB_64, KIB_64).unwrap());
        let address = memory.address();
        // A 16x16 RGBA8 texture placed there that shaders write.
        // SAFETY: a description is 56 bytes, 14 words.
        let mut resource: [u32; 14] = unsafe { std::mem::transmute_copy(&target(16, 16, 28)) };
        const UNORDERED: u32 = 4;
        resource[12] = UNORDERED;
        // Format, 2D (an unordered access view's 4), mip 0, plane 0; then
        // the address.
        let mut uav = [0u32; 16];
        uav[..2].copy_from_slice(&[28, 4]);
        (uav[10], uav[11]) = (address as u32, (address >> 32) as u32);
        let heap = crate::descriptor::TestHeap::views(4);
        let title = crate::descriptor::heap(heap.0);
        let host = title.host().expect("a host heap");
        let handle = title.start + 32;
        crate::descriptor::create_placed_uav(std::ptr::null(), &resource, &uav, handle);
        let kept = |h: &HostHeap, i: usize| {
            h.kept.lock().unwrap_or_else(|e| e.into_inner())[i]
                .as_ref()
                .map(|t| t.as_raw() as usize)
        };
        let host_of = |r: This| crate::resource::host_resource(r).map(|t| t.as_raw() as usize);
        // The texture made for the memory, resident before the slot is made
        // (see `resident`).
        let written = crate::resource::texture_at(address, &resource).unwrap();
        resident(written);
        assert!(matches!(
            host.mirror(1, handle, true),
            Some(View::Unordered(..))
        ));
        assert_eq!(
            crate::resource::texture_at(address, &resource),
            Some(written)
        );
        assert!(host_of(written).is_some());
        assert_eq!(
            kept(host, 1),
            host_of(written),
            "the slot views its texture"
        );
        // A shader view placed over the same memory, the title's
        // description without unordered access: the GPU-written copy.
        let mut sampled = resource;
        sampled[12] = 0;
        let mut srv = [0u32; 16];
        srv[..6].copy_from_slice(&[28, 4, 0x1688, 0, 0, 1]);
        (srv[10], srv[11]) = (uav[10], uav[11]);
        let other = title.start + 2 * 32;
        crate::descriptor::create_placed_srv(std::ptr::null(), &sampled, &srv, other);
        assert!(matches!(
            host.mirror(2, other, true),
            Some(View::Shader(..))
        ));
        assert_eq!(kept(host, 2), host_of(written), "one host copy");
        drop(heap);
        crate::resource::memory_remapped(address as usize, KIB_64 as usize);
        drop(std::mem::ManuallyDrop::into_inner(memory));
    }

    /// A buffer view of title buffer `r` as the title describes one:
    /// `count` elements from element `first`, in `format` (0 with a
    /// `stride` for a structured view; typeless, 39, with the raw flag for
    /// a raw one); unordered access or a shader resource view.
    fn buffer_view(
        unordered: bool,
        r: &crate::testgpu::TitleBuffer,
        (format, stride, raw): (u32, u32, bool),
        (first, count): (u32, u32),
    ) -> View {
        let mut w = [0u32; 10];
        (w[0], w[1]) = (format, 1);
        let flags = u32::from(raw);
        if unordered {
            // First element (64 bits), count, stride, counter offset, flags.
            (w[2], w[4], w[5], w[8]) = (first, count, stride, flags);
            View::Unordered(r.resource() as usize, 0, Some(w))
        } else {
            // The default mapping, first element, count, stride, flags.
            w[2] = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            (w[4], w[6], w[7], w[8]) = (first, count, stride, flags);
            View::Shader(r.resource() as usize, Some(w))
        }
    }

    #[test]
    fn three_byte_views_keep_their_title_range_in_both_directions() {
        let mut resource = [0; 14];
        resource[0] = 1;
        for format in 118..=120 {
            for unordered in [false, true] {
                let mut words = [0; 16];
                words[0] = format;
                words[1] = 1;
                words[10] = 4096;
                let first = if unordered { 2 } else { 4 };
                words[first] = 4;
                words[first + 2] = 7;
                assert_eq!(
                    buffer_range(&View::Placed(unordered, resource, words)),
                    Some((4108, 21))
                );
            }
        }
    }

    const RAW: (u32, u32, bool) = (39, 0, true);
    const fn structured(stride: u32) -> (u32, u32, bool) {
        (0, stride, false)
    }

    /// A view heap with a runtime stride table: an entry per slot, written
    /// with the slot's descriptor. A raw view is read in its own slot; a
    /// structured or typed one through a raw view of the same bytes in the
    /// pool, which starts up to 12 bytes early; a null or constant view,
    /// or a slot given back, holds none.
    #[test]
    fn stride_table_entries_follow_the_views() {
        use crate::testgpu::TitleBuffer;
        let test = "descriptor::host::tests::stride_table_entries_follow_the_views";
        let Some(_gpu) = crate::testgpu::device(test) else {
            return;
        };
        let heap = HostHeap::new_with(false, 8, true).expect("a host view heap");
        let none = HostHeap::new(false, 8).expect("a host view heap");
        assert!(heap.stride_table().is_some() && none.stride_table().is_none());
        let buffer = TitleBuffer::new(1 << 12);
        let pool = heap.alias_base();
        let start = buffer.shadow().offset(buffer.address());
        let bias = |byte: u64| ((start + byte) % 16) as u32;
        let entry = |stride, raw_slot, bias, count| StrideEntry {
            stride,
            raw_slot: Some(raw_slot),
            bias,
            count,
        };
        heap.make_at(0, &buffer_view(false, &buffer, RAW, (0, 64)));
        assert_eq!(heap.entry(0), entry(0, 0, 0, 64), "raw: its own slot");
        heap.make_at(1, &buffer_view(false, &buffer, structured(12), (3, 5)));
        assert_eq!(heap.entry(1), entry(12, pool, bias(36), 5));
        heap.make_at(2, &buffer_view(true, &buffer, structured(20), (1, 4)));
        assert_eq!(heap.entry(2), entry(20, pool + 1, bias(20), 4));
        // Typed RGBA32F: 16-byte elements.
        heap.make_at(3, &buffer_view(false, &buffer, (2, 0, false), (2, 8)));
        assert_eq!(heap.entry(3), entry(16, pool + 2, bias(32), 8));
        // The same view again keeps its raw view.
        heap.make_at(1, &buffer_view(false, &buffer, structured(12), (3, 5)));
        assert_eq!(heap.entry(1), entry(12, pool, bias(36), 5));
        // Nothing to read: null, a constant view.
        heap.make_null(2);
        assert_eq!(heap.entry(2), StrideEntry::NONE);
        heap.constant_at(3, buffer.shadow().gpu(buffer.address()), 256);
        assert_eq!(heap.entry(3), StrideEntry::NONE);
        // A dynamic slot given back loses its entry before it is free.
        let first = heap.dynamic(2).expect("dynamic slots");
        heap.make_at(
            first + 1,
            &buffer_view(false, &buffer, structured(16), (0, 4)),
        );
        assert!(heap.entry(first + 1).raw_slot.is_some());
        heap.release(first, 2);
        assert_eq!(heap.entry(first + 1), StrideEntry::NONE);
    }

    /// A view past its host copy's end is cut to it (the host refuses, and
    /// may remove the device for, a view past its resource), one of no
    /// elements there or of a stride the host has no structured buffers for
    /// is null; the entry counts what the cut view holds.
    #[test]
    fn buffer_views_are_cut_to_their_host_copy() {
        use crate::testgpu::TitleBuffer;
        let test = "descriptor::host::tests::buffer_views_are_cut_to_their_host_copy";
        let Some(_gpu) = crate::testgpu::device(test) else {
            return;
        };
        let heap = HostHeap::new_with(false, 8, true).expect("a host view heap");
        let buffer = TitleBuffer::new(1 << 16);
        let copy = buffer.shadow();
        // SAFETY: a live buffer.
        let width = unsafe { copy.buffer.GetDesc() }.Width;
        let first = (width - copy.offset(buffer.address())) / 16 - 2;
        // Two elements of 16 bytes are left from `first`.
        assert!(heap.make_at(
            0,
            &buffer_view(false, &buffer, structured(16), (first as u32, 100))
        ));
        assert_eq!(heap.entry(0).count, 2);
        let past = (width / 16) as u32;
        assert!(!heap.make_at(1, &buffer_view(false, &buffer, structured(16), (past, 4))));
        assert_eq!(heap.entry(1), StrideEntry::NONE);
        assert!(!heap.make_at(2, &buffer_view(false, &buffer, structured(4100), (0, 1))));
        assert!(!heap.make_at(3, &buffer_view(false, &buffer, structured(16), (0, 0))));
        // A typeless view without the raw flag is no view the host makes.
        assert!(!heap.make_at(4, &buffer_view(false, &buffer, (39, 0, false), (0, 4))));
    }

    /// `xtp-shader`'s stride tests' program, translated for a target with
    /// the table: it loads `[2, 4]` of the view in table slot 0 (no stride
    /// known), stores it at `[1, 8]` of the unordered view in slot 1 and
    /// adds 10 twice at `[2, 0]` there.
    fn unknown_stride_program() -> xtp_shader::emit::HostShader {
        use xtp_shader::native::{ComputeIo, StageIo};
        use xtp_shader::{HwStage, NativeProgram, ResourceType, RootParam, RootType, UsedResource};
        let words: &[u32] = &[
            0xBE85_0380, // s_mov_b32 s5, 0
            0xF408_0202,
            0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], 0x0
            0xF408_0302,
            0xFA00_0020, // s_load_dwordx4 s[12:15], s[4:5], 0x20
            0x7E00_0282, // v_mov_b32 v0, 2
            0x7E02_0284, // v_mov_b32 v1, 4
            0xE030_3000,
            0x8002_0200, // buffer_load_dword v2, v[0:1], s[8:11], 0 idxen offen
            0x7E00_0281, // v_mov_b32 v0, 1
            0x7E02_0288, // v_mov_b32 v1, 8
            0xE070_3000,
            0x8003_0200, // buffer_store_dword v2, v[0:1], s[12:15], 0 idxen offen
            0x7E00_0282, // v_mov_b32 v0, 2
            0x7E02_0280, // v_mov_b32 v1, 0
            0x7E06_028A, // v_mov_b32 v3, 10
            0xE0C8_3000,
            0x8003_0300, // buffer_atomic_add v3, v[0:1], s[12:15], 0 idxen offen
            0xE0C8_3000,
            0x8003_0300, // buffer_atomic_add v3, v[0:1], s[12:15], 0 idxen offen
            0xBF81_0000, // s_endpgm
        ];
        let used = |kind, table_offset_dwords| UsedResource {
            kind,
            root: RootType::Table,
            table_offset_dwords,
            lower_bound: 0,
            stride: 0,
            dwords: 4,
            user_data: 4,
            slot: 1,
        };
        let param = |root, sgprs, user_data, slot| RootParam {
            root,
            table_offset_dwords: 0,
            sgprs,
            base_register: 0,
            user_data,
            slot,
        };
        let native = NativeProgram {
            stage: HwStage::Cs,
            wave64: false,
            float_mode: 0,
            vgprs: 4,
            layout: vec![
                param(RootType::Constants, 4, 0, 0),
                param(RootType::Table, 1, 4, 1),
            ],
            used: vec![used(ResourceType::Srv, 0), used(ResourceType::Uav, 8)],
            io: StageIo::Cs(ComputeIo {
                rsrc2: 5 << 1,
                threads: [1, 1, 1],
            }),
            code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        let p = xtp_shader::ir::build(&native.code, native.wave64, native.vgprs).unwrap();
        let st = xtp_shader::structure::structure(&p).unwrap();
        let res = xtp_shader::resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
        let io = xtp_shader::io::raise(&p, &native).unwrap();
        // An empty container (no signature parts): the tag, a digest,
        // version 1, its size, no parts.
        let mut container = b"DXBC".to_vec();
        container.extend([0; 16]);
        container.extend([1u32, 32, 0].into_iter().flat_map(u32::to_le_bytes));
        let c = xtp_shader::Container::parse(&container).unwrap();
        let target = xtp_shader::emit::Target {
            stride_table: true,
            ..xtp_shader::emit::Target::ANY
        };
        let cs = xtp_shader::emit::emit_for(&c, &native, &p, &st, &io, &res, target).unwrap();
        assert!(cs.stride_table);
        cs
    }

    /// The root signature and pipeline of `cs` as `pipeline::host` makes
    /// them: user data as root constants (parameter 0), the stride table as
    /// a root SRV (parameter 1), the heaps indexed directly.
    fn table_pipeline(
        device: &ID3D12Device,
        cs: &xtp_shader::emit::HostShader,
    ) -> (ID3D12RootSignature, ID3D12PipelineState) {
        use windows::Win32::Graphics::Direct3D::ID3DBlob;
        let params = [
            D3D12_ROOT_PARAMETER1 {
                ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
                Anonymous: D3D12_ROOT_PARAMETER1_0 {
                    Constants: D3D12_ROOT_CONSTANTS {
                        ShaderRegister: 0,
                        RegisterSpace: xtp_shader::emit::USER_DATA_SPACE,
                        Num32BitValues: cs.user_data_dwords(),
                    },
                },
                ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
            },
            D3D12_ROOT_PARAMETER1 {
                ParameterType: D3D12_ROOT_PARAMETER_TYPE_SRV,
                Anonymous: D3D12_ROOT_PARAMETER1_0 {
                    Descriptor: D3D12_ROOT_DESCRIPTOR1 {
                        ShaderRegister: 0,
                        RegisterSpace: xtp_shader::emit::STRIDE_SPACE,
                        Flags: D3D12_ROOT_DESCRIPTOR_FLAG_DATA_VOLATILE,
                    },
                },
                ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
            },
        ];
        let desc = D3D12_VERSIONED_ROOT_SIGNATURE_DESC {
            Version: D3D_ROOT_SIGNATURE_VERSION_1_1,
            Anonymous: D3D12_VERSIONED_ROOT_SIGNATURE_DESC_0 {
                Desc_1_1: D3D12_ROOT_SIGNATURE_DESC1 {
                    NumParameters: params.len() as u32,
                    pParameters: params.as_ptr(),
                    NumStaticSamplers: 0,
                    pStaticSamplers: std::ptr::null(),
                    Flags: D3D12_ROOT_SIGNATURE_FLAG_CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED
                        | D3D12_ROOT_SIGNATURE_FLAG_SAMPLER_HEAP_DIRECTLY_INDEXED,
                },
            },
        };
        // SAFETY: the descriptions point at live data; the blob is read
        // before it goes.
        unsafe {
            let mut blob: Option<ID3DBlob> = None;
            D3D12SerializeVersionedRootSignature(&desc, &mut blob, None).unwrap();
            let blob = blob.unwrap();
            let bytes = std::slice::from_raw_parts(
                blob.GetBufferPointer().cast::<u8>(),
                blob.GetBufferSize(),
            );
            let root: ID3D12RootSignature = device.CreateRootSignature(0, bytes).unwrap();
            let pso = device
                .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                    pRootSignature: std::mem::transmute_copy(&root),
                    CS: D3D12_SHADER_BYTECODE {
                        pShaderBytecode: cs.dxil.as_ptr().cast(),
                        BytecodeLength: cs.dxil.len(),
                    },
                    ..Default::default()
                })
                .unwrap();
            (root, pso)
        }
    }

    /// A translated program reads and writes title buffers through a real
    /// host heap's stride table, whatever view of 4, 12, 16 or 20 bytes an
    /// element (structured, typed, also from an element off 16 bytes of the
    /// copy) or raw the title put in the slots: the bytes the console
    /// reads and writes, the next element never (with 4-byte elements the
    /// load at 4 and the store at 8 are past the element: 0, dropped).
    #[test]
    fn a_translated_program_reads_and_writes_through_the_heaps_table() {
        use crate::testgpu::{TitleBuffer, TitleList};
        let test = "descriptor::host::tests::a_translated_program_reads_and_writes_through_the_heaps_table";
        let Some(gpu) = crate::testgpu::device(test) else {
            return;
        };
        let host = gpu.host();
        let heap = HostHeap::new_with(false, 8, true).expect("a host view heap");
        let cs = unknown_stride_program();
        let (root, pso) = table_pipeline(&host.device, &cs);
        let (table, _) = heap.stride_table().expect("a table");
        let (src, dst) = (TitleBuffer::new(1 << 12), TitleBuffer::new(1 << 12));
        src.set_words(|i| 100 + i as u32);
        let data = src.words();
        for (kind, shape, stride, first, count) in [
            ("structured 4", structured(4), 4, 0, 64),
            ("structured 12", structured(12), 12, 0, 21),
            ("structured 16", structured(16), 16, 0, 16),
            ("structured 20", structured(20), 20, 0, 12),
            ("typed RGBA32F", (2, 0, false), 16, 0, 16),
            ("structured 12 from element 3", structured(12), 12, 3, 10),
            ("structured 20 from element 3", structured(20), 20, 3, 6),
            ("raw", RAW, 0, 0, 64),
        ] {
            dst.set_words(|_| 0);
            heap.make_at(0, &buffer_view(false, &src, shape, (first, count)));
            heap.make_at(1, &buffer_view(true, &dst, shape, (first, count)));
            assert!(heap.entry(0).count > 0 && heap.entry(1).count > 0, "{kind}");
            let list = TitleList::new();
            let l = list.host();
            let (from, to) = (src.shadow(), dst.shadow());
            from.refresh(l, src.address(), 1 << 12);
            to.refresh(l, dst.address(), 1 << 12);
            to.gpu_writes(l, dst.address(), 1 << 12);
            l.keep(&root);
            l.keep(&pso);
            let zeros = vec![0u32; cs.user_data_dwords() as usize];
            // SAFETY: an open list with every root argument set; the heap
            // and its table outlive the run.
            unsafe {
                l.list
                    .SetDescriptorHeaps(&crate::descriptor::heaps_with_own_samplers(Some(
                        &heap.heap,
                    )));
                l.list.SetComputeRootSignature(&root);
                l.list.SetPipelineState(&pso);
                l.list.SetComputeRoot32BitConstants(
                    0,
                    zeros.len() as u32,
                    zeros.as_ptr().cast(),
                    0,
                );
                l.list.SetComputeRootShaderResourceView(1, table);
                l.list.Dispatch(1, 1, 1);
            }
            list.run();
            let out = dst.words();
            let at = |index: u32, off: u32| (((first + index) * stride + off) / 4) as usize;
            let inside = |off: u32| stride == 0 || off + 4 <= stride;
            let mut want = vec![0; 1 << 10];
            if inside(8) {
                want[at(1, 8)] = if inside(4) { data[at(2, 4)] } else { 0 };
            }
            want[at(2, 0)] = 20;
            assert_eq!(out[..64], want[..64], "{kind}");
            assert!(out[64..].iter().all(|&w| w == 0), "{kind}: nothing else");
        }
        // SAFETY: a plain device query.
        assert!(unsafe { host.device.GetDeviceRemovedReason() }.is_ok());
    }

    /// What a list running [`unknown_stride_program`] refreshes of the
    /// title memory the program reads.
    #[derive(Clone, Copy)]
    enum Reads<'a> {
        /// The source buffer's first 4 KiB.
        Whole,
        /// Each run of a listing.
        Runs(&'a Picked),
        /// Nothing itself: the step of the list's picks does
        /// ([`PickedUse::defer`]).
        Step,
    }

    /// Runs [`unknown_stride_program`] once over `heap` with the table at
    /// user data `base` (the byte offset the host gives a table: slot 0 is
    /// read at `base`, slot 1 written at `base + 32`), `src`'s copy
    /// refreshed for it (`reads`) and `dst`'s written back; what `dst`
    /// holds after.
    fn run_table_program(
        heap: &HostHeap,
        pipeline: &(ID3D12RootSignature, ID3D12PipelineState),
        cs: &xtp_shader::emit::HostShader,
        (src, dst): (&crate::testgpu::TitleBuffer, &crate::testgpu::TitleBuffer),
        base: u32,
        reads: Reads<'_>,
    ) -> Vec<u32> {
        let list = crate::testgpu::TitleList::new();
        record_table_program(list.host(), heap, pipeline, cs, (src, dst), base, reads);
        list.run();
        dst.words()
    }

    /// Records into `l` what [`run_table_program`] runs.
    fn record_table_program(
        l: &crate::host::HostList,
        heap: &HostHeap,
        pipeline: &(ID3D12RootSignature, ID3D12PipelineState),
        cs: &xtp_shader::emit::HostShader,
        (src, dst): (&crate::testgpu::TitleBuffer, &crate::testgpu::TitleBuffer),
        base: u32,
        reads: Reads<'_>,
    ) {
        let (root, pso) = pipeline;
        let (table, _) = heap.stride_table().expect("a table");
        let (from, to) = (src.shadow(), dst.shadow());
        match reads {
            Reads::Runs(picked) => {
                for (copy, at, end) in &picked.reads() {
                    copy.refresh(l, *at, end - at);
                }
            }
            Reads::Whole => from.refresh(l, src.address(), 1 << 12),
            Reads::Step => {}
        }
        to.refresh(l, dst.address(), 1 << 12);
        to.gpu_writes(l, dst.address(), 1 << 12);
        l.keep(root);
        l.keep(pso);
        let mut values = vec![0u32; cs.user_data_dwords() as usize];
        for u in &cs.user_data {
            if u.root == xtp_shader::RootType::Table {
                values[u.index as usize] = base;
            }
        }
        // SAFETY: an open list with every root argument set; the heap and
        // its table outlive the run.
        unsafe {
            l.list
                .SetDescriptorHeaps(&crate::descriptor::heaps_with_own_samplers(Some(
                    &heap.heap,
                )));
            l.list.SetComputeRootSignature(root);
            l.list.SetPipelineState(pso);
            l.list
                .SetComputeRoot32BitConstants(0, values.len() as u32, values.as_ptr().cast(), 0);
            l.list.SetComputeRootShaderResourceView(1, table);
            l.list.Dispatch(1, 1, 1);
        }
    }

    /// Codex review of 54ab53d, P1: the table is a root SRV, so a shader
    /// reading the entry of whatever slot it computes read past the table,
    /// or, for a title slot past a capped host heap, the entry of another
    /// table in the dynamic slots there. The header bounds it: a title slot
    /// is read only below the slots the heap mirrors, a slot the host
    /// remapped (its user data marked, `emit::REMAPPED_SLOT`) below the
    /// table's entries, anything else as no view (loads 0, stores dropped).
    #[test]
    fn stride_table_reads_stay_inside_the_table_and_tell_remapped_slots() {
        use crate::testgpu::TitleBuffer;
        let test = "descriptor::host::tests::stride_table_reads_stay_inside_the_table_and_tell_remapped_slots";
        let Some(gpu) = crate::testgpu::device(test) else {
            return;
        };
        let host = gpu.host();
        let heap = HostHeap::new_with(false, 8, true).expect("a host view heap");
        let len = heap.strides.as_ref().expect("a table").table.len;
        assert_eq!(heap.table_header(), (8, len), "mirrored, entries");
        let cs = unknown_stride_program();
        let pipeline = table_pipeline(&host.device, &cs);
        let (src, dst) = (TitleBuffer::new(1 << 12), TitleBuffer::new(1 << 12));
        src.set_words(|i| 100 + i as u32);
        // The same views in title slots 0 and 1 and in the first two
        // dynamic slots (8 and 9), as a table made afresh holds them.
        let views = [
            buffer_view(false, &src, structured(16), (0, 16)),
            buffer_view(true, &dst, structured(16), (0, 16)),
        ];
        let first = heap.dynamic(2).expect("dynamic slots");
        assert_eq!(first, 8);
        for (k, view) in views.iter().enumerate() {
            heap.make_at(k as u32, view);
            heap.make_at(first + k as u32, view);
        }
        let mut want = vec![0u32; 1 << 10];
        // [1, 8] of 16-byte elements: dword 6; [2, 4]: dword 9; [2, 0]: 8.
        (want[6], want[8]) = (src.words()[9], 20);
        let mark = xtp_shader::emit::REMAPPED_SLOT << 5;
        for (what, base, reads) in [
            ("title slots 0 and 1", 0, true),
            ("the remapped table in slots 8 and 9", (8 * 32) | mark, true),
            ("title slot 8, past the mirrored ones", 8 * 32, false),
            ("a title slot far past the table", 1 << 30, false),
            (
                "a remapped slot past the table",
                ((1 << 25) * 32) | mark,
                false,
            ),
        ] {
            dst.set_words(|_| 0);
            let out = run_table_program(&heap, &pipeline, &cs, (&src, &dst), base, Reads::Whole);
            if reads {
                assert_eq!(out[..64], want[..64], "{what}");
            } else {
                assert!(
                    out.iter().all(|&w| w == 0),
                    "{what}: nothing read or written"
                );
            }
        }
        heap.release(first, 2);
        // SAFETY: a plain device query.
        assert!(unsafe { host.device.GetDeviceRemovedReason() }.is_ok());
    }

    /// The pipeline of a kernel that copies the 64 dwords it reads through
    /// a one-view table (`t0`, raw) to the root UAV `u0`.
    fn raw_copy_pipeline(host: &crate::host::Host) -> (ID3D12RootSignature, ID3D12PipelineState) {
        use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
        use windows::Win32::Graphics::Direct3D::ID3DBlob;
        use windows::core::s;
        const SOURCE: &str = r#"
#define RS "DescriptorTable(SRV(t0)), UAV(u0)"
ByteAddressBuffer src : register(t0);
RWByteAddressBuffer dst : register(u0);
[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    dst.Store(id.x * 4, src.Load(id.x * 4));
}
"#;
        let mut code: Option<ID3DBlob> = None;
        // SAFETY: the source and names live through the call.
        unsafe {
            D3DCompile(
                SOURCE.as_ptr().cast(),
                SOURCE.len(),
                s!("raw copy test"),
                None,
                None,
                s!("main"),
                s!("cs_5_1"),
                0,
                0,
                &mut code,
                None,
            )
        }
        .expect("the test kernel compiles");
        let code = code.expect("compiled code");
        // SAFETY: the blob's own bytes, which hold the root signature.
        let bytes = unsafe {
            std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize())
        };
        // SAFETY: plain object creation from live bytes.
        unsafe {
            let root: ID3D12RootSignature = host.device.CreateRootSignature(0, bytes).unwrap();
            let pso: ID3D12PipelineState = host
                .device
                .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                    pRootSignature: std::mem::transmute_copy(&root),
                    CS: D3D12_SHADER_BYTECODE {
                        pShaderBytecode: bytes.as_ptr().cast(),
                        BytecodeLength: bytes.len(),
                    },
                    ..Default::default()
                })
                .unwrap();
            (root, pso)
        }
    }

    /// Records into `l` a copy of the 64 dwords pool slot `slot` of `heap`
    /// reads into `dst`, refreshing `src`'s copy and writing `dst` back.
    fn copy_pool_slot(
        l: &crate::host::HostList,
        heap: &HostHeap,
        (root, pso): &(ID3D12RootSignature, ID3D12PipelineState),
        slot: u32,
        (src, dst): (&crate::testgpu::TitleBuffer, &crate::testgpu::TitleBuffer),
    ) {
        let (from, to) = (src.shadow(), dst.shadow());
        from.refresh(l, src.address(), 1 << 12);
        to.refresh(l, dst.address(), 256);
        to.gpu_writes(l, dst.address(), 256);
        l.keep(root);
        l.keep(pso);
        // SAFETY: an open list; live objects kept until it is reset; the
        // slot is in the heap set on it.
        unsafe {
            l.list
                .SetDescriptorHeaps(&crate::descriptor::heaps_with_own_samplers(Some(
                    &heap.heap,
                )));
            l.list.SetComputeRootSignature(root);
            l.list.SetPipelineState(pso);
            l.list
                .SetComputeRootDescriptorTable(0, heap.gpu_handle(slot));
            l.list
                .SetComputeRootUnorderedAccessView(1, to.gpu(dst.address()));
            l.list.Dispatch(1, 1, 1);
        }
    }

    /// Codex review of 54ab53d, P1: a slot whose view changes window while
    /// a submission the GPU may be running read the table gets another raw
    /// view; the one that submission may read keeps its window (rewritten
    /// in place before, it gave such a submission the new window with the
    /// old stride and bias). Held in flight on WARP behind a fence: a list
    /// in flight when the view changes, reading the old raw view, still
    /// reads the old window. Once the GPU has run what was in flight, the
    /// raw view comes back to the pool, made null, with its host buffer:
    /// the retiring ones are bounded by the submissions in flight (before,
    /// only taking a raw view emptied them, and a slot rewritten in place
    /// kept a buffer for good). A list handed to the queue after the
    /// change reads the entry as it is after (the table is read when the
    /// GPU runs), so it is not waited for: here the reader was submitted
    /// after the change until the Codex review of 244cf78 (finding 3),
    /// which gives raw views back as soon as the fence reaches what was in
    /// flight, not at the heap's next use. A view whose window stays keeps
    /// its raw view.
    #[test]
    fn raw_views_a_submission_in_flight_may_read_stay_until_it_has_run() {
        use crate::testgpu::{TitleBuffer, pattern};
        let _serial = crate::host::submissions::serial();
        let test = "descriptor::host::tests::raw_views_a_submission_in_flight_may_read_stay_until_it_has_run";
        let Some(gpu) = crate::testgpu::device(test) else {
            return;
        };
        let host = gpu.host();
        let heap = HostHeap::with_pool(8, 4).expect("a host view heap");
        let pipeline = raw_copy_pipeline(host);
        let (src, dst) = (TitleBuffer::new(1 << 12), TitleBuffer::new(1 << 12));
        src.set_words(pattern);
        assert_eq!(src.shadow().offset(src.address()), 0, "its own copy");
        let pool = heap.alias_base();
        let raw_slot = |i| heap.entry(i).raw_slot.expect("a raw view");
        // Bytes 0 to 96: 8 elements of 12.
        heap.make_at(0, &buffer_view(false, &src, structured(12), (0, 8)));
        assert_eq!(raw_slot(0), pool);
        heap.make_at(0, &buffer_view(false, &src, structured(12), (0, 8)));
        assert_eq!(raw_slot(0), pool, "the same window: the same raw view");
        assert_eq!(heap.pool_state(), (0, 3));
        // Submissions in flight, held on the GPU, one of which reads the raw
        // view the entry names now (as a shader that read the entry before
        // the change does).
        let mut gate = crate::host::Gate::close(host);
        let running = host.new_list().expect("a host list");
        running.close();
        host.execute(&[&running]);
        assert!(running.running());
        dst.set_words(|_| 0xdead_beef);
        let reader = host.new_list().expect("a host list");
        copy_pool_slot(&reader, &heap, &pipeline, pool, (&src, &dst));
        reader.close();
        host.execute(&[&reader]);
        // Bytes 32 to 96 as 4 elements of 16: another raw view.
        heap.make_at(0, &buffer_view(false, &src, structured(16), (2, 4)));
        assert_eq!(raw_slot(0), pool + 1);
        assert_eq!(heap.entry(0).stride, 16);
        assert_eq!(heap.pool_state(), (1, 2), "the old one retires");
        // Changes go on while they are in flight, each to another raw view.
        heap.make_at(0, &buffer_view(false, &src, structured(20), (0, 4)));
        assert_eq!(heap.pool_state(), (2, 1));
        gate.open();
        reader.settle();
        let old: Vec<u32> = (0..64)
            .map(|i| if i < 24 { pattern(i) } else { 0 })
            .collect();
        assert_eq!(dst.words()[..64], old[..], "the old window, untouched");
        // The GPU has run them: back, null, once the host has finished
        // them, the heap untouched meanwhile.
        completed_until(host, "never given back", || heap.pool_state() == (0, 3));
        assert!(heap.stride_hold().is_some());
        assert_eq!(heap.pool_state(), (0, 3));
        let list = crate::testgpu::TitleList::new();
        copy_pool_slot(list.host(), &heap, &pipeline, pool, (&src, &dst));
        list.run();
        assert!(dst.words()[..64].iter().all(|&w| w == 0), "made null");
        // SAFETY: a plain device query.
        assert!(unsafe { host.device.GetDeviceRemovedReason() }.is_ok());
    }

    /// Waits until the GPU has run what was handed to the queue so far,
    /// other tests' submissions too (one handed over but not yet in the
    /// timeline is not waited for by a drain). Waits for the GPU fence
    /// only, not for host write-back: call `HostList::settle` before
    /// reading back what a submission wrote.
    fn gpu_caught_up(host: &crate::host::Host) {
        let ticket = host.executing();
        let started = std::time::Instant::now();
        while host.completed() < ticket {
            assert!(started.elapsed().as_secs() < 30, "the GPU never got there");
            host.drain();
            std::thread::yield_now();
        }
    }

    /// Codex review of 54ab53d, P2: a view the full pool gives no raw view
    /// reads as none, but was marked made, so the next read of the same
    /// record found nothing to do and the slot read zeros through the table
    /// for good. Now it is made again at the next read, and gets a raw view
    /// once one has come back.
    #[test]
    fn a_full_pool_gives_raw_views_again_once_one_comes_back() {
        use crate::testgpu::TitleBuffer;
        let test = "descriptor::host::tests::a_full_pool_gives_raw_views_again_once_one_comes_back";
        let Some(gpu) = crate::testgpu::device(test) else {
            return;
        };
        let heap = HostHeap::with_pool(8, 2).expect("a host view heap");
        let buffer = TitleBuffer::new(1 << 12);
        // Title descriptor memory of 8 slots.
        let slots = vec![0u64; 8 * 4];
        let at = |i: u64| slots.as_ptr() as u64 + 32 * i;
        for i in 0..3u32 {
            let View::Shader(r, Some(words)) = buffer_view(false, &buffer, structured(12), (i, 4))
            else {
                unreachable!()
            };
            crate::descriptor::create_srv(
                std::ptr::null(),
                r as crate::object::This,
                &words,
                at(u64::from(i)),
            );
        }
        for i in 0..3u32 {
            assert!(heap.mirror(i, at(u64::from(i)), true).is_some());
        }
        assert!(heap.entry(0).raw_slot.is_some() && heap.entry(1).raw_slot.is_some());
        assert_eq!(heap.entry(2), StrideEntry::NONE, "the pool is full");
        assert_eq!(heap.made.lock().unwrap().keys[2], 0, "to be made again");
        heap.mirror(2, at(2), true);
        assert_eq!(heap.entry(2), StrideEntry::NONE, "still full");
        // Slot 0 holds nothing any more: its raw view comes back once the
        // GPU has run what was in flight (other tests' submissions), and
        // slot 2's next read takes it.
        heap.make_null(0);
        gpu_caught_up(gpu.host());
        heap.mirror(2, at(2), true);
        let entry = heap.entry(2);
        assert_eq!((entry.stride, entry.count), (12, 4));
        assert_eq!(entry.raw_slot, Some(heap.alias_base()));
    }

    /// Codex review of 54ab53d, P1: a sweep read a slot's record from title
    /// memory without a lock while another thread published a new one there
    /// and made the slot; the sweep then made the slot from the record it
    /// had read, which stayed. The record is read again under the slot's
    /// lock: the one made is the one published.
    #[test]
    fn a_record_published_during_a_sweep_is_the_one_made() {
        use crate::testgpu::TitleBuffer;
        let test = "descriptor::host::tests::a_record_published_during_a_sweep_is_the_one_made";
        let Some(_gpu) = crate::testgpu::device(test) else {
            return;
        };
        let heap = HostHeap::new_with(false, 8, true).expect("a host view heap");
        let buffer = TitleBuffer::new(1 << 12);
        let slots = vec![0u64; 8 * 4];
        let handle = slots.as_ptr() as u64 + 2 * 32;
        let srv = |stride| {
            let View::Shader(r, Some(words)) =
                buffer_view(false, &buffer, structured(stride), (0, 4))
            else {
                unreachable!()
            };
            (r as crate::object::This, words)
        };
        let (old, new) = (srv(12), srv(20));
        crate::descriptor::create_srv(std::ptr::null(), old.0, &old.1, handle);
        // Between the sweep's read and its lock, the title publishes the new
        // record and makes the slot as its view creation does.
        let at = std::ptr::from_ref(&heap) as usize;
        BETWEEN.with(|b| {
            *b.borrow_mut() = Some(Box::new(move || {
                // SAFETY: called inside `heap.mirror` below, while it lives.
                let heap = unsafe { &*(at as *const HostHeap) };
                crate::descriptor::create_srv(std::ptr::null(), new.0, &new.1, handle);
                heap.mirror(2, handle, false);
            }))
        });
        heap.mirror(2, handle, true);
        assert!(BETWEEN.with(|b| b.borrow().is_none()), "the publisher ran");
        assert_eq!(heap.entry(2).stride, 20, "the record published");
        let [_, i, generation] = header_at(handle);
        let key = u64::from(generation) << 32 | u64::from(i);
        assert_eq!(heap.made.lock().unwrap().keys[2], key);
    }

    /// Codex review of 54ab53d, P1: a view of 12-byte elements from byte 20
    /// of its copy has a raw view in the table but a null slot, and was left
    /// out of what sweeps refresh, so a shader that picked it at run time
    /// read a stale copy; a view cut to its copy kept its full length and,
    /// past the title buffer, was not found at all. Both are listed, as cut
    /// to the copy, and the program reads what the title wrote last.
    #[test]
    fn views_read_through_the_table_only_are_refreshed_for_picks() {
        use crate::testgpu::TitleBuffer;
        let test =
            "descriptor::host::tests::views_read_through_the_table_only_are_refreshed_for_picks";
        let Some(gpu) = crate::testgpu::device(test) else {
            return;
        };
        RUN_TIME_VIEWS_TEST.set(Some(true));
        let host = gpu.host();
        let heap = HostHeap::new_with(false, 8, true).expect("a host view heap");
        let cs = unknown_stride_program();
        let pipeline = table_pipeline(&host.device, &cs);
        let (src, dst) = (TitleBuffer::new(1 << 16), TitleBuffer::new(1 << 12));
        let base = src.address();
        assert_eq!(src.shadow().offset(base), 0, "its own copy");
        // A placed buffer view (a buffer resource description, 64 KiB) of
        // structured 12-byte elements from `at`: `count` of them.
        let placed = |at: u64, count: u32, width: u32| {
            let mut resource = [0u32; 14];
            (resource[0], resource[4]) = (1, width);
            let mut desc = [0u32; 16];
            (desc[1], desc[2], desc[6], desc[7]) =
                (1, D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING, count, 12);
            (desc[10], desc[11]) = (at as u32, (at >> 32) as u32);
            (resource, desc)
        };
        let slots = vec![0u64; 8 * 4];
        let at = |i: u64| slots.as_ptr() as u64 + 32 * i;
        let (r, d) = placed(base + 20, 4, (1 << 16) - 20);
        crate::descriptor::create_placed_srv(std::ptr::null(), &r, &d, at(0));
        // From 24 bytes before the copy's end, 100 elements: two fit.
        let (r, d) = placed(base + (1 << 16) - 24, 100, 24);
        crate::descriptor::create_placed_srv(std::ptr::null(), &r, &d, at(2));
        // The program's output, in slot 1 (made by a draw's table).
        heap.make_at(1, &buffer_view(true, &dst, structured(12), (0, 16)));
        for i in [0, 2] {
            heap.mirror(i, at(i.into()), false);
        }
        assert_eq!(heap.waiting(), 2, "both wait for a sweep");
        let picked = heap.picked_spans(slots.as_ptr() as u64).expect("a listing");
        assert_eq!(heap.waiting(), 0);
        assert_eq!(heap.entry(0).stride, 12, "the table names a raw view");
        let reads: Vec<(u64, u64)> = picked.reads().iter().map(|&(_, a, b)| (a, b)).collect();
        assert_eq!(
            reads,
            [(base + 20, base + (1 << 16))],
            "both, cut to the copy"
        );
        // The title writes the buffer; the list that picks refreshes what
        // the listing says, and only that.
        src.set_words(|i| 7000 + i as u32);
        dst.set_words(|_| 0);
        let out = run_table_program(&heap, &pipeline, &cs, (&src, &dst), 0, Reads::Runs(&picked));
        // [2, 4] of the view from byte 20: byte 48, dword 12, stored at
        // [1, 8] of the output's 12-byte elements (dword 5); 20 at [2, 0].
        assert_eq!((out[5], out[6]), (7012, 20), "what the title wrote last");
        RUN_TIME_VIEWS_TEST.set(None);
    }

    /// Without `XTP_RUN_TIME_VIEWS=1`, nothing is kept for shaders that pick
    /// slots at run time, as before the switch: the title's buffer views do
    /// not wait for a sweep, and there is no listing.
    #[test]
    fn nothing_waits_for_sweeps_with_the_switch_off() {
        use crate::testgpu::TitleBuffer;
        let test = "descriptor::host::tests::nothing_waits_for_sweeps_with_the_switch_off";
        let Some(_gpu) = crate::testgpu::device(test) else {
            return;
        };
        RUN_TIME_VIEWS_TEST.set(Some(false));
        let heap = HostHeap::new_with(false, 8, true).expect("a host view heap");
        let buffer = TitleBuffer::new(1 << 12);
        let slots = vec![0u64; 8 * 4];
        let View::Shader(r, Some(words)) = buffer_view(false, &buffer, structured(16), (0, 4))
        else {
            unreachable!()
        };
        let handle = slots.as_ptr() as u64;
        crate::descriptor::create_srv(std::ptr::null(), r as crate::object::This, &words, handle);
        heap.mirror(0, handle, false);
        assert_eq!(heap.waiting(), 0);
        assert!(heap.picked_spans(handle).is_none());
        heap.mirror(0, handle, true);
        assert!(heap.made.lock().unwrap().buffers.is_empty());
        RUN_TIME_VIEWS_TEST.set(None);
    }

    /// Codex review of 54ab53d, P2: a title description whose range cannot
    /// be represented (a first element of 2^62 elements of 16 bytes) is no
    /// range, where it overflowed through an `extern "system"` call.
    #[test]
    fn unrepresentable_view_ranges_are_refused() {
        assert_eq!(range_of(0x1000, 2, 4, 16), Some((0x1020, 64)));
        assert_eq!(range_of(0x1000, 1 << 62, 4, 16), None);
        assert_eq!(range_of(u64::MAX - 8, 1, 1, 16), None);
        assert_eq!(range_of(u64::MAX - 64, 0, 8, 16), None);
        let mut words = [0u32; 10];
        // A structured view of 16 bytes from element 2^62.
        (words[1], words[4], words[5], words[6], words[7]) = (1, 0, 1 << 30, 4, 16);
        let placed = View::Placed(false, [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], {
            let mut d = [0u32; 16];
            d[..10].copy_from_slice(&words);
            d[10] = 0x1000;
            d
        });
        assert_eq!(buffer_range(&placed), None);
    }

    /// A root SRV's one-view table (VIctory #493): a shader reading 64
    /// words through a raw view of 30 bytes from byte 16 of a host copy
    /// gets words 4 to 11 (the partial last one whole) and zeros after
    /// them, although the copy holds data there.
    #[test]
    fn root_srv_views_read_zeros_past_their_end() {
        use crate::testgpu::{TitleBuffer, TitleList, pattern};
        use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
        use windows::Win32::Graphics::Direct3D::ID3DBlob;
        use windows::core::s;
        const SOURCE: &str = r#"
#define RS "DescriptorTable(SRV(t0)), UAV(u0)"
ByteAddressBuffer src : register(t0);
RWByteAddressBuffer dst : register(u0);
[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    dst.Store(id.x * 4, src.Load(id.x * 4));
}
"#;
        let test = "descriptor::host::tests::root_srv_views_read_zeros_past_their_end";
        let Some(gpu) = crate::testgpu::device(test) else {
            return;
        };
        let host = gpu.host();
        let heap = HostHeap::new(false, 4).expect("a host view heap");
        let mut code: Option<ID3DBlob> = None;
        // SAFETY: the source and names live through the call.
        unsafe {
            D3DCompile(
                SOURCE.as_ptr().cast(),
                SOURCE.len(),
                s!("root srv test"),
                None,
                None,
                s!("main"),
                s!("cs_5_1"),
                0,
                0,
                &mut code,
                None,
            )
        }
        .expect("the test kernel compiles");
        let code = code.expect("compiled code");
        // SAFETY: the blob's own bytes, which hold the root signature.
        let bytes = unsafe {
            std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize())
        };
        // SAFETY: plain object creation from live bytes.
        let (root, pso) = unsafe {
            let root: ID3D12RootSignature = host.device.CreateRootSignature(0, bytes).unwrap();
            let pso: ID3D12PipelineState = host
                .device
                .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                    pRootSignature: std::mem::transmute_copy(&root),
                    CS: D3D12_SHADER_BYTECODE {
                        pShaderBytecode: bytes.as_ptr().cast(),
                        BytecodeLength: bytes.len(),
                    },
                    ..Default::default()
                })
                .unwrap();
            (root, pso)
        };

        let src = TitleBuffer::new(1 << 16);
        src.set_words(pattern);
        let dst = TitleBuffer::new(1 << 16);
        dst.set_words(|_| 0xdead_beef);
        let (from, to) = (src.shadow(), dst.shadow());
        let slot = heap.dynamic(1).expect("a dynamic slot");
        heap.raw_at(slot, &from.buffer, from.offset(src.address()) + 16, 30);

        let list = TitleList::new();
        let l = list.host();
        from.refresh(l, src.address(), 256);
        to.refresh(l, dst.address(), 256);
        to.gpu_writes(l, dst.address(), 256);
        l.keep(&root);
        l.keep(&pso);
        // SAFETY: an open list; live objects kept until it is reset; the
        // slot is in the heap set on it.
        unsafe {
            l.list
                .SetDescriptorHeaps(&crate::descriptor::heaps_with_own_samplers(Some(
                    &heap.heap,
                )));
            l.list.SetComputeRootSignature(&root);
            l.list.SetPipelineState(&pso);
            l.list
                .SetComputeRootDescriptorTable(0, heap.gpu_handle(slot));
            l.list
                .SetComputeRootUnorderedAccessView(1, to.gpu(dst.address()));
            l.list.Dispatch(1, 1, 1);
        }
        list.run();
        let words = dst.words();
        let wanted: Vec<u32> = (0..64)
            .map(|i| if i < 8 { pattern(i + 4) } else { 0 })
            .collect();
        assert_eq!(words[..64], wanted[..]);
        // SAFETY: a plain device query.
        assert!(unsafe { host.device.GetDeviceRemovedReason() }.is_ok());
    }

    /// A texel of 0x80 in an RGBA8 texture, read on the GPU through a view
    /// made as the title's are ([`fits`]): an sRGB or linear view gets its
    /// own conversion whichever of the two the texture is, and a UINT view
    /// of a UNORM texture reads the byte, where the host casts; without the
    /// cast each view reads in the texture's format, as before (the sRGB
    /// view of UNORM then read 0.502). Ported from fxceless's VIctory fork
    /// (#644, by laptopnotebook), after akimfomin's proposal in #578.
    #[test]
    fn views_read_in_their_own_format_where_the_host_casts() {
        use crate::resource::format::{Caps, HostCaps};
        use crate::testgpu::{TitleBuffer, TitleList};
        use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
        use windows::Win32::Graphics::Direct3D::ID3DBlob;
        use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;
        use windows::core::s;
        const SOURCE: &str = r#"
#define RS "DescriptorTable(SRV(t0, numDescriptors = 2)), UAV(u0)"
Texture2D<float4> floats : register(t0);
Texture2D<uint4> integers : register(t1);
RWByteAddressBuffer dst : register(u0);
[RootSignature(RS)]
[numthreads(1, 1, 1)]
void main() {
    dst.Store(0, asuint(floats.Load(int3(0, 0, 0)).r));
    dst.Store(4, integers.Load(int3(0, 0, 0)).r);
}
"#;
        const UNORM: u32 = 28;
        const SRGB: u32 = 29;
        const UINT: u32 = 30;
        // 0x80 read as linear, and as sRGB converted to linear.
        const LINEAR: f32 = 128.0 / 255.0;
        const DECODED: f32 = 0.215_860_5;
        let test = "descriptor::host::tests::views_read_in_their_own_format_where_the_host_casts";
        let Some(gpu) = crate::testgpu::device(test) else {
            return;
        };
        let host = gpu.host();
        let device = &host.device;
        let casting = HostCaps.fully_typed_casting();
        eprintln!("{test}: the host casts fully typed formats: {casting}");
        let mut code: Option<ID3DBlob> = None;
        // SAFETY: the source and names live through the call.
        unsafe {
            D3DCompile(
                SOURCE.as_ptr().cast(),
                SOURCE.len(),
                s!("view cast test"),
                None,
                None,
                s!("main"),
                s!("cs_5_1"),
                0,
                0,
                &mut code,
                None,
            )
        }
        .expect("the test kernel compiles");
        let code = code.expect("compiled code");
        // SAFETY: the blob's own bytes, which hold the root signature.
        let bytes = unsafe {
            std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize())
        };
        // SAFETY: plain object creation from live bytes.
        let (root, pso) = unsafe {
            let root: ID3D12RootSignature = device.CreateRootSignature(0, bytes).unwrap();
            let pso: ID3D12PipelineState = device
                .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                    pRootSignature: std::mem::transmute_copy(&root),
                    CS: D3D12_SHADER_BYTECODE {
                        pShaderBytecode: bytes.as_ptr().cast(),
                        BytecodeLength: bytes.len(),
                    },
                    ..Default::default()
                })
                .unwrap();
            (root, pso)
        };
        let committed = |desc: &D3D12_RESOURCE_DESC, kind, state| {
            let heap = D3D12_HEAP_PROPERTIES {
                Type: kind,
                ..Default::default()
            };
            let mut made: Option<ID3D12Resource> = None;
            // SAFETY: plain creation of a resource of the test's own.
            unsafe {
                device
                    .CreateCommittedResource(
                        &heap,
                        D3D12_HEAP_FLAG_NONE,
                        desc,
                        state,
                        None,
                        &mut made,
                    )
                    .unwrap();
            }
            made.unwrap()
        };
        // The texel's bytes, uploaded through a buffer (rows 256 apart).
        let upload = committed(
            &D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: 256,
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                ..Default::default()
            },
            D3D12_HEAP_TYPE_UPLOAD,
            D3D12_RESOURCE_STATE_GENERIC_READ,
        );
        let mut mapped = std::ptr::null_mut();
        // SAFETY: the mapping holds the buffer's 256 bytes until unmapped.
        unsafe {
            upload.Map(0, None, Some(&mut mapped)).unwrap();
            mapped.cast::<u8>().write_bytes(0x80, 4);
            upload.Unmap(0, None);
        }
        let heap = HostHeap::new(false, 4).expect("a host view heap");
        let slots = heap.dynamic(2).expect("two dynamic slots");
        let dst = TitleBuffer::new(1 << 16);
        // The texture's format, then the view's.
        for (texture_format, view_format) in [
            (UNORM, UNORM),
            (UNORM, SRGB),
            (SRGB, UNORM),
            (SRGB, SRGB),
            (UNORM, UINT),
        ] {
            let read_as = if casting { view_format } else { texture_format };
            if view_format == UINT && !casting {
                // A UNORM view where the shader reads integers: no answer.
                continue;
            }
            let texture = committed(
                &D3D12_RESOURCE_DESC {
                    Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                    Width: 1,
                    Height: 1,
                    DepthOrArraySize: 1,
                    MipLevels: 1,
                    Format: DXGI_FORMAT(texture_format as i32),
                    SampleDesc: DXGI_SAMPLE_DESC {
                        Count: 1,
                        Quality: 0,
                    },
                    ..Default::default()
                },
                D3D12_HEAP_TYPE_DEFAULT,
                D3D12_RESOURCE_STATE_COPY_DEST,
            );
            let mut view = D3D12_SHADER_RESOURCE_VIEW_DESC {
                Format: DXGI_FORMAT(view_format as i32),
                ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
                Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                    Texture2D: D3D12_TEX2D_SRV {
                        MipLevels: 1,
                        ..Default::default()
                    },
                },
            };
            assert!(fits(&texture, Some(&mut view.Format), false));
            assert_eq!(
                view.Format.0 as u32, read_as,
                "{texture_format} as {view_format}"
            );
            // The view in the slot of the shader's type, a null view of
            // that type in the other.
            let (slot, other, null_format) = if view_format == UINT {
                (slots + 1, slots, DXGI_FORMAT_R8G8B8A8_UNORM)
            } else {
                (slots, slots + 1, DXGI_FORMAT_R8G8B8A8_UINT)
            };
            let null = D3D12_SHADER_RESOURCE_VIEW_DESC {
                Format: null_format,
                ..view
            };
            // SAFETY: a live texture and slots of our heap; a null resource
            // is allowed with a description.
            unsafe {
                device.CreateShaderResourceView(&texture, Some(&view), heap.cpu(slot));
                device.CreateShaderResourceView(
                    None::<&ID3D12Resource>,
                    Some(&null),
                    heap.cpu(other),
                );
            }
            dst.set_words(|_| 0xdead_beef);
            let to = dst.shadow();
            let list = TitleList::new();
            let l = list.host();
            to.refresh(l, dst.address(), 256);
            to.gpu_writes(l, dst.address(), 256);
            l.keep(&root);
            l.keep(&pso);
            l.keep(&texture);
            l.keep(&upload);
            let into = D3D12_TEXTURE_COPY_LOCATION {
                // SAFETY: a borrowed pointer the location never releases.
                pResource: unsafe { std::mem::transmute_copy(&texture) },
                Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    SubresourceIndex: 0,
                },
            };
            let from = D3D12_TEXTURE_COPY_LOCATION {
                // SAFETY: as above.
                pResource: unsafe { std::mem::transmute_copy(&upload) },
                Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                        Offset: 0,
                        Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                            Format: DXGI_FORMAT(texture_format as i32),
                            Width: 1,
                            Height: 1,
                            Depth: 1,
                            RowPitch: 256,
                        },
                    },
                },
            };
            // SAFETY: an open list; live objects kept until it is reset; the
            // slots are in the heap set on it.
            unsafe {
                l.list.CopyTextureRegion(&into, 0, 0, 0, &from, None);
                l.list.ResourceBarrier(&[crate::host::transition(
                    &texture,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                )]);
                l.list
                    .SetDescriptorHeaps(&crate::descriptor::heaps_with_own_samplers(Some(
                        &heap.heap,
                    )));
                l.list.SetComputeRootSignature(&root);
                l.list.SetPipelineState(&pso);
                l.list
                    .SetComputeRootDescriptorTable(0, heap.gpu_handle(slots));
                l.list
                    .SetComputeRootUnorderedAccessView(1, to.gpu(dst.address()));
                l.list.Dispatch(1, 1, 1);
            }
            list.run();
            let words = dst.words();
            if view_format == UINT {
                assert_eq!(words[1], 0x80, "{texture_format} as {view_format}");
                continue;
            }
            let red = f32::from_bits(words[0]);
            let wanted = if read_as == SRGB { DECODED } else { LINEAR };
            assert!(
                (red - wanted).abs() < 0.002,
                "{texture_format} as {view_format} (cast: {casting}): red {red}, wanted {wanted}"
            );
        }
        // SAFETY: a plain device query.
        assert!(unsafe { device.GetDeviceRemovedReason() }.is_ok());
    }

    /// A shader view the title made without a description, of a texture
    /// whose host copy is typeless (depth that shaders read), is made in
    /// the format that reads its depth; it was null.
    #[test]
    fn whole_views_of_typeless_copies_are_made() {
        use crate::resource::tests_support::{committed, depth_unordered};
        let test = "descriptor::host::tests::whole_views_of_typeless_copies_are_made";
        let Some(_gpu) = crate::testgpu::device(test) else {
            return;
        };
        const D32_FLOAT: u32 = 40;
        let depth = committed(&depth_unordered(16, 16, D32_FLOAT));
        let started = std::time::Instant::now();
        let copy = loop {
            if let Some(copy) = crate::resource::host_resource(depth) {
                break copy;
            }
            assert!(started.elapsed().as_secs() < 5, "never resident");
            std::thread::sleep(std::time::Duration::from_millis(1));
        };
        // SAFETY: a live resource.
        let format = unsafe { copy.GetDesc() }.Format;
        assert_eq!(format, DXGI_FORMAT_R32_TYPELESS);
        let whole = crate::resource::whole_typed_view(depth).expect("a description");
        assert_eq!(whole.Format, DXGI_FORMAT_R32_FLOAT);
        assert_eq!(whole.ViewDimension, D3D12_SRV_DIMENSION_TEXTURE2D);
        let heap = HostHeap::new(false, 4).expect("a host view heap");
        let slot = heap.dynamic(1).expect("a dynamic slot");
        assert!(heap.make_at(slot, &View::Shader(depth as usize, None)));
        // A typed copy is viewed by D3D12's own whole view.
        let colour = committed(&crate::resource::tests_support::sampled(16, 16, 28));
        let started = std::time::Instant::now();
        while crate::resource::host_resource(colour).is_none() {
            assert!(started.elapsed().as_secs() < 5, "never resident");
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        assert!(crate::resource::whole_typed_view(colour).is_none());
        drop(heap);
        type Release = extern "system" fn(crate::object::This) -> u32;
        for r in [depth, colour] {
            crate::object::slot::<Release>(r, 0x10)(r);
        }
    }

    /// Codex review of 244cf78, finding 7: a sweep read a slot's record
    /// header (12 bytes of title memory) while another thread wrote it:
    /// view creation wrote under the views' lock only, and descriptor
    /// copies took none. Every access takes one lock now. Two threads on
    /// WARP: one writes two records into a slot in turn, by descriptor
    /// copies and by view creation; the other reads the slot's header and
    /// makes the host slot from it, again and again. Every header read is
    /// one of the two records whole (their view indices and generations
    /// differ, so half of one and half of the other is neither, and names
    /// no view), and every slot made is one of the two views.
    #[test]
    fn record_headers_are_read_whole_while_another_thread_writes_them() {
        use crate::testgpu::TitleBuffer;
        const WRITES: u32 = 200_000;
        let test = "descriptor::host::tests::record_headers_are_read_whole_while_another_thread_writes_them";
        let Some(_gpu) = crate::testgpu::device(test) else {
            return;
        };
        let heap = HostHeap::new_with(false, 8, true).expect("a host view heap");
        let buffer = TitleBuffer::new(1 << 12);
        // Title descriptor memory: the two records, the slot, scratch.
        let slots = vec![0u64; 8 * 4];
        let at = |i: u64| slots.as_ptr() as u64 + 32 * i;
        let (a, b, slot, scratch) = (at(0), at(1), at(2), at(5));
        let null = std::ptr::null();
        let srv = |buffer: &TitleBuffer, stride: u32| {
            let View::Shader(r, Some(words)) =
                buffer_view(false, buffer, structured(stride), (0, 4))
            else {
                unreachable!()
            };
            (r, words)
        };
        // Record B in an entry freed once (a later generation), A in
        // another: every pair of their halves is neither.
        let gone = TitleBuffer::new(1 << 12);
        let (r, words) = srv(&gone, 16);
        crate::descriptor::create_srv(null, r as crate::object::This, &words, scratch);
        crate::descriptor::forget(r);
        let record_b = srv(&buffer, 20);
        crate::descriptor::create_srv(null, record_b.0 as crate::object::This, &record_b.1, b);
        let hb = crate::descriptor::header(b);
        let record_a = (1..64)
            .map(|k| srv(&buffer, 20 + 4 * k))
            .find(|&(r, words)| {
                crate::descriptor::create_srv(null, r as crate::object::This, &words, a);
                let ha = crate::descriptor::header(a);
                ha[1] != hb[1] && ha[2] != hb[2]
            })
            .expect("two records of other indices and generations");
        let ha = crate::descriptor::header(a);
        let views = [a, b].map(|h| crate::descriptor::view(h).expect("a view"));
        crate::descriptor::copy_simple(null, 1, slot, a, 0);
        let done = std::sync::atomic::AtomicBool::new(false);
        let (mut reads, mut made) = (0u64, 0u64);
        std::thread::scope(|s| {
            s.spawn(|| {
                for n in 0..WRITES {
                    let (from, (r, words)) = if n % 2 == 0 {
                        (b, record_b)
                    } else {
                        (a, record_a)
                    };
                    if n % 3 == 0 {
                        let r = r as crate::object::This;
                        crate::descriptor::create_srv(std::ptr::null(), r, &words, slot);
                    } else {
                        crate::descriptor::copy_simple(std::ptr::null(), 1, slot, from, 0);
                    }
                }
                done.store(true, Ordering::SeqCst);
            });
            while !done.load(Ordering::SeqCst) {
                for _ in 0..16 {
                    let h = crate::descriptor::header(slot);
                    assert!(
                        h == ha || h == hb,
                        "a torn header {h:x?}: {ha:x?} or {hb:x?}"
                    );
                    reads += 1;
                }
                match heap.mirror_once(2, slot, true, true) {
                    Ok(Some(view)) => {
                        assert!(views.contains(&view), "a slot made from no record");
                        made += 1;
                    }
                    Ok(None) => panic!("a header that names no view was made"),
                    // Published between the two reads: read again.
                    Err(_) => {}
                }
            }
        });
        assert!(
            reads > 0 && made > 0,
            "the reader ran ({reads} reads, {made} made)"
        );
        eprintln!("{reads} header reads, {made} slots made while {WRITES} records were written");
    }

    /// Codex review of 244cf78, finding 3: a raw view that retired behind a
    /// submission came back to the pool only at the heap's next use, after
    /// the GPU had run it: a heap left alone kept it, and its host buffer,
    /// for good. The host's completion of that submission gives it back
    /// now, made null; nothing here touches the heap after the change.
    #[test]
    fn retired_raw_views_go_back_when_the_fence_reaches_them() {
        use crate::testgpu::TitleBuffer;
        let _serial = crate::host::submissions::serial();
        let test = "descriptor::host::tests::retired_raw_views_go_back_when_the_fence_reaches_them";
        let Some(gpu) = crate::testgpu::device(test) else {
            return;
        };
        let host = gpu.host();
        let heap = HostHeap::with_pool(8, 4).expect("a host view heap");
        let src = TitleBuffer::new(1 << 12);
        heap.make_at(0, &buffer_view(false, &src, structured(12), (0, 8)));
        assert_eq!(heap.pool_state(), (0, 3));
        let mut gate = crate::host::Gate::close(host);
        let running = host.new_list().expect("a host list");
        running.close();
        host.execute(&[&running]);
        heap.make_at(0, &buffer_view(false, &src, structured(16), (2, 4)));
        assert_eq!(heap.pool_state(), (1, 2), "the old one retires");
        gate.open();
        let s = heap.strides.as_ref().expect("a table");
        completed_until(host, "never given back", || heap.pool_state() == (0, 3));
        completed_until(host, "never off the fence's list", || !s.pool().listed);
        // SAFETY: a plain device query.
        assert!(unsafe { host.device.GetDeviceRemovedReason() }.is_ok());
    }

    /// Waits until the GPU has caught up ([`gpu_caught_up`]), then until
    /// `done` holds, which the host's completion of submissions brings:
    /// without finishing submissions itself meanwhile (a test that drains
    /// in a loop finishes other tests' submissions, whose own drain then
    /// returns before their writes are back in title memory).
    fn completed_until(host: &crate::host::Host, what: &str, done: impl Fn() -> bool) {
        gpu_caught_up(host);
        let started = std::time::Instant::now();
        while !done() {
            assert!(started.elapsed().as_secs() < 30, "{what}");
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
    }

    /// Codex review of 244cf78, finding 4: in a table made afresh, a view
    /// the full pool gave no raw view kept an entry of none, which the
    /// list's later draws of the same table read (`draw.rs` keeps the table
    /// for the list), after one came back too: they read zeros. The slot
    /// gets the raw view that comes back, here when the host fence reaches
    /// it (nothing else touches the heap), before the list that reads the
    /// table is submitted: its dispatch reads and writes through it.
    #[test]
    fn a_remapped_slot_gets_a_raw_view_once_one_comes_back() {
        use crate::testgpu::{TitleBuffer, TitleList};
        let _serial = crate::host::submissions::serial();
        let test = "descriptor::host::tests::a_remapped_slot_gets_a_raw_view_once_one_comes_back";
        let Some(gpu) = crate::testgpu::device(test) else {
            return;
        };
        let host = gpu.host();
        let heap = HostHeap::with_pool(8, 2).expect("a host view heap");
        let cs = unknown_stride_program();
        let pipeline = table_pipeline(&host.device, &cs);
        let (src, dst, other) = (
            TitleBuffer::new(1 << 12),
            TitleBuffer::new(1 << 12),
            TitleBuffer::new(1 << 12),
        );
        src.set_words(|i| 100 + i as u32);
        // A dynamic slot holds one of the pool's two raw views; the table
        // made afresh after it takes the other for its shader view, and
        // none is left for its unordered one.
        let first = heap.dynamic(3).expect("dynamic slots");
        let (table, holder) = (first, first + 2);
        heap.make_at(holder, &buffer_view(false, &other, structured(16), (0, 16)));
        heap.make_at(table, &buffer_view(false, &src, structured(16), (0, 16)));
        heap.make_at(table + 1, &buffer_view(true, &dst, structured(16), (0, 16)));
        assert_eq!(heap.entry(table + 1), StrideEntry::NONE, "the pool is full");
        // The list that reads the table, recorded now and submitted after.
        dst.set_words(|_| 0);
        let list = TitleList::new();
        let base = (table * 32) | (xtp_shader::emit::REMAPPED_SLOT << 5);
        record_table_program(
            list.host(),
            &heap,
            &pipeline,
            &cs,
            (&src, &dst),
            base,
            Reads::Whole,
        );
        // The holder's view goes while a submission is in flight: its raw
        // view comes back once the fence reaches that submission.
        let mut gate = crate::host::Gate::close(host);
        let running = host.new_list().expect("a host list");
        running.close();
        host.execute(&[&running]);
        heap.make_null(holder);
        assert_eq!(heap.entry(table + 1), StrideEntry::NONE, "none free yet");
        gate.open();
        completed_until(host, "never given one", || {
            heap.entry(table + 1).raw_slot.is_some()
        });
        let entry = heap.entry(table + 1);
        assert_eq!((entry.stride, entry.count), (16, 16));
        list.run();
        // [2, 4] of 16-byte elements (dword 9) stored at [1, 8] (dword 6);
        // 10 added twice at [2, 0] (dword 8).
        let out = dst.words();
        assert_eq!((out[6], out[8]), (src.words()[9], 20));
        heap.release(first, 3);
        // SAFETY: a plain device query.
        assert!(unsafe { host.device.GetDeviceRemovedReason() }.is_ok());
    }

    /// Codex review of 244cf78, finding 6: sweeps made host copies for 8 ms
    /// a frame and left the views still waiting null, so a dispatch of that
    /// frame that picked one read 0. Every waiting view is made now, past
    /// the frame's time too: here the time is spent before the sweep (as a
    /// first copy that took it all would), and the views of buffers whose
    /// host copy is not made yet are made all the same.
    #[test]
    fn every_waiting_view_is_made_past_the_frames_time() {
        use crate::testgpu::TitleBuffer;
        let test = "descriptor::host::tests::every_waiting_view_is_made_past_the_frames_time";
        let Some(_gpu) = crate::testgpu::device(test) else {
            return;
        };
        RUN_TIME_VIEWS_TEST.set(Some(true));
        let heap = HostHeap::new_with(false, 8, true).expect("a host view heap");
        let buffers: Vec<TitleBuffer> = (0..3).map(|_| TitleBuffer::new(1 << 12)).collect();
        let slots = vec![0u64; 8 * 4];
        let at = |i: usize| slots.as_ptr() as u64 + 32 * i as u64;
        for (i, b) in buffers.iter().enumerate() {
            let View::Shader(r, Some(words)) = buffer_view(false, b, structured(16), (0, 16))
            else {
                unreachable!()
            };
            let r = r as crate::object::This;
            crate::descriptor::create_srv(std::ptr::null(), r, &words, at(i));
            heap.mirror(i as u32, at(i), false);
        }
        assert_eq!(heap.waiting(), 3);
        let copied = |b: &TitleBuffer| crate::shadow::existing(b.address(), 1 << 12).is_some();
        assert!(!buffers.iter().any(copied), "no host copy yet");
        heap.spend_picked_time();
        let picked = heap.picked_spans(at(0)).expect("a listing");
        assert_eq!(heap.waiting(), 0, "none left for later");
        assert!(buffers.iter().all(copied));
        for i in 0..3 {
            assert_eq!(heap.entry(i).stride, 16, "slot {i} made");
        }
        assert_eq!(picked.reads().len(), 3);
        RUN_TIME_VIEWS_TEST.set(None);
    }

    /// The step of a list's picks ([`PickedUse::defer`]; VIctory 6d021fe):
    /// N buffers whose unordered views the title writes before each of N
    /// dispatches that write through slots they pick add one step to the
    /// list and look at no copy while recording (Codex review of 8612e27,
    /// P2: each new recording caught up on the listing from nothing, a
    /// refresh step per run). At submission the first list's step is given
    /// every run, a later list's step of the same submission what changed
    /// since (nothing if nothing did), every run again after a step that
    /// wrote title memory, and a new submission every run; runs are
    /// declared written once per submission. Each copy that changed is
    /// looked at once (Codex reviews of 244cf78 and a090f22, P2: N(N+1)/2
    /// copies over the N dispatches). A copy no view reads any more is
    /// given no more.
    #[test]
    fn picks_add_one_step_and_submissions_are_given_the_listing_once() {
        use crate::testgpu::{TitleBuffer, TitleList};
        const N: usize = 8;
        let test = "descriptor::host::tests::picks_add_one_step_and_submissions_are_given_the_listing_once";
        let Some(_gpu) = crate::testgpu::device(test) else {
            return;
        };
        RUN_TIME_VIEWS_TEST.set(Some(true));
        let heap = HostHeap::new_with(false, 16, true).expect("a host view heap");
        let buffers: Vec<TitleBuffer> = (0..=N).map(|_| TitleBuffer::new(1 << 12)).collect();
        let slots = vec![0u64; 16 * 4];
        let at = |i: usize| slots.as_ptr() as u64 + 32 * i as u64;
        let uav = |i: usize| {
            let View::Unordered(r, _, Some(words)) =
                buffer_view(true, &buffers[i], structured(16), (0, 16))
            else {
                unreachable!()
            };
            let r = r as crate::object::This;
            crate::descriptor::create_uav(std::ptr::null(), r, std::ptr::null(), &words, at(i));
            heap.mirror(i as u32, at(i), false);
        };
        let list = TitleList::new();
        let l = list.host();
        let mut used = PickedUse::default();
        for i in 0..N {
            uav(i);
            let picked = heap.picked_spans(at(0)).expect("a listing");
            assert_eq!(picked.writes().len(), i + 1);
            used.defer(l, &picked, true);
        }
        assert_eq!(l.steps(), 1, "one step for the N picks");
        assert_eq!(heap.visits(), 0, "no copy looked at while recording");
        let step = used.step.as_ref().map(|(_, s)| s.clone()).expect("a step");
        let count = |(reads, writes): (Vec<Span>, Vec<Span>)| (reads.len(), writes.len());
        let submission = crate::shadow::submission();
        assert_eq!(
            count(step.due(submission, 0)),
            (N, N),
            "every run, declared"
        );
        assert_eq!(heap.visits(), 2 * N as u64, "each copy once, for each");
        // The submission's next list.
        assert_eq!(count(step.due(submission, 0)), (0, 0), "nothing changed");
        assert_eq!(heap.visits(), 2 * N as u64, "nothing looked at");
        // A view the title writes meanwhile: its run alone.
        uav(N);
        heap.picked_spans(at(0)).expect("a listing");
        assert_eq!(count(step.due(submission, 0)), (1, 1), "what changed");
        assert_eq!(heap.visits(), 2 * N as u64 + 2);
        // After a step of the submission that wrote title memory.
        assert_eq!(
            count(step.due(submission, 1)),
            (N + 1, 0),
            "refreshed again"
        );
        // What the submission's GPU work touches: once.
        let (reads, writes) = step.touches(submission);
        assert_eq!((reads.len(), writes.len()), (2 * N + 2, N + 1));
        let (reads, writes) = step.touches(submission);
        assert!(reads.is_empty() && writes.is_empty(), "taken");
        // The next submission: every run.
        let next = crate::shadow::submission();
        assert_eq!(count(step.due(next, 0)), (N + 1, N + 1));
        // The title copies an empty slot over slot 0: its copy goes from
        // the listing, and from what submissions are given.
        crate::descriptor::copy_simple(std::ptr::null(), 1, at(0), at(15), 0);
        heap.mirror(0, at(0), true);
        assert_eq!(count(step.due(crate::shadow::submission(), 0)), (N, N));
        let held = heap.listed.listing().due.reads.len();
        assert_eq!(held, N, "the copy no view reads is not held");
        RUN_TIME_VIEWS_TEST.set(None);
    }

    /// Codex review of 8612e27, P2: N disjoint views of one host copy, each
    /// written before a pick, made each pick copy the copy's whole map of
    /// runs: 1 + ... + N runs looked at under the listing's lock. A pick
    /// looks at nothing now, and the submission is given the N runs once,
    /// the copy looked at once.
    #[test]
    fn disjoint_views_of_one_copy_are_given_at_submission_not_at_each_pick() {
        use crate::testgpu::{TitleBuffer, TitleList};
        const N: usize = 8;
        let test = "descriptor::host::tests::disjoint_views_of_one_copy_are_given_at_submission_not_at_each_pick";
        let Some(_gpu) = crate::testgpu::device(test) else {
            return;
        };
        RUN_TIME_VIEWS_TEST.set(Some(true));
        let heap = HostHeap::new_with(false, 16, true).expect("a host view heap");
        // Views two pages apart: a run each.
        let apart = 2 * RUN_GAP as usize;
        let buffer = TitleBuffer::new(N * apart);
        let slots = vec![0u64; 16 * 4];
        let at = |i: usize| slots.as_ptr() as u64 + 32 * i as u64;
        let list = TitleList::new();
        let l = list.host();
        let mut used = PickedUse::default();
        for i in 0..N {
            let first = (i * apart / 16) as u32;
            let View::Shader(r, Some(words)) =
                buffer_view(false, &buffer, structured(16), (first, 16))
            else {
                unreachable!()
            };
            let r = r as crate::object::This;
            crate::descriptor::create_srv(std::ptr::null(), r, &words, at(i));
            heap.mirror(i as u32, at(i), false);
            let picked = heap.picked_spans(at(0)).expect("a listing");
            used.defer(l, &picked, false);
        }
        assert_eq!((l.steps(), heap.visits()), (1, 0));
        let step = used.step.as_ref().map(|(_, s)| s.clone()).expect("a step");
        let (reads, writes) = step.due(crate::shadow::submission(), 0);
        let base = buffer.address();
        let runs: Vec<(u64, u64)> = reads.iter().map(|&(_, a, b)| (a - base, b - a)).collect();
        let want: Vec<(u64, u64)> = (0..N).map(|i| ((i * apart) as u64, 256)).collect();
        assert_eq!(runs, want, "the N runs, by address");
        assert!(writes.is_empty(), "no shader writes through them");
        assert_eq!(heap.visits(), 1, "the copy looked at once");
        RUN_TIME_VIEWS_TEST.set(None);
    }

    /// Codex review of 8612e27, P2 (a new bug): a closed list submitted
    /// again after the title pointed the slot it picks at another buffer
    /// added that buffer's refresh to its steps at each submission, and
    /// held them, and the buffers, until its reset: after N submissions, N
    /// refreshes for one slot. The list keeps its one step; each
    /// submission is given what the slot holds then, and the listing holds
    /// that buffer's copy alone.
    #[test]
    fn a_resubmitted_list_refreshes_what_its_slot_holds_now_only() {
        use crate::testgpu::{TitleBuffer, TitleList};
        const N: usize = 4;
        let test =
            "descriptor::host::tests::a_resubmitted_list_refreshes_what_its_slot_holds_now_only";
        let Some(gpu) = crate::testgpu::device(test) else {
            return;
        };
        RUN_TIME_VIEWS_TEST.set(Some(true));
        let host = gpu.host();
        let heap = HostHeap::new_with(false, 8, true).expect("a host view heap");
        let slots = vec![0u64; 8 * 4];
        let start = slots.as_ptr() as u64;
        let buffers: Vec<TitleBuffer> = (0..N).map(|_| TitleBuffer::new(1 << 12)).collect();
        let write = |b: &TitleBuffer| {
            let View::Shader(r, Some(words)) = buffer_view(false, b, structured(16), (0, 16))
            else {
                unreachable!()
            };
            crate::descriptor::create_srv(
                std::ptr::null(),
                r as crate::object::This,
                &words,
                start,
            );
            heap.mirror(0, start, false);
        };
        write(&buffers[0]);
        let list = TitleList::new();
        let l = list.host();
        let used = Arc::new(Mutex::new(PickedUse::default()));
        let picked = heap.picked_spans(start).expect("a listing");
        used.lock().unwrap().defer(l, &picked, false);
        // The list sweeps at each of its submissions, as `draw.rs` has it.
        let (at, sweeping) = (std::ptr::from_ref(&heap) as usize, used.clone());
        l.at_submit(move |list| {
            // SAFETY: the heap outlives the list's submissions below.
            let heap = unsafe { &*(at as *const HostHeap) };
            if let Some(picked) = heap.picked_spans(start) {
                sweeping.lock().unwrap().defer(list, &picked, false);
            }
        });
        type Close = extern "system" fn(crate::object::This) -> crate::object::HRESULT;
        let close = crate::object::slot::<Close>(list.this(), 0x48);
        assert_eq!(close(list.this()), crate::object::S_OK);
        for (k, b) in buffers.iter().enumerate() {
            if k > 0 {
                write(b);
            }
            host.submit(&[l]);
            host.drain();
            assert_eq!(l.steps(), 1, "submission {k}: still the one step");
            let copies: Vec<usize> = heap.listed.listing().due.reads.keys().copied().collect();
            let now = Arc::as_ptr(&b.shadow()) as usize;
            assert_eq!(
                copies,
                [now],
                "submission {k}: the buffer the slot holds alone"
            );
        }
        RUN_TIME_VIEWS_TEST.set(None);
        // SAFETY: a plain device query.
        assert!(unsafe { host.device.GetDeviceRemovedReason() }.is_ok());
    }

    /// Two lists of one submission pick slots of one heap: the first one's
    /// step refreshes what the listing reads, the second's nothing more
    /// (VIctory 6d021fe), and the programs of both read what the title
    /// wrote last; written again, the next submission of the two reads the
    /// new bytes.
    #[test]
    fn lists_of_a_submission_share_one_refresh_and_read_the_titles_bytes() {
        use crate::testgpu::{TitleBuffer, TitleList};
        let test = "descriptor::host::tests::lists_of_a_submission_share_one_refresh_and_read_the_titles_bytes";
        let Some(gpu) = crate::testgpu::device(test) else {
            return;
        };
        RUN_TIME_VIEWS_TEST.set(Some(true));
        let host = gpu.host();
        let heap = HostHeap::new_with(false, 8, true).expect("a host view heap");
        let cs = unknown_stride_program();
        let pipeline = table_pipeline(&host.device, &cs);
        let src = TitleBuffer::new(1 << 12);
        let dsts = [TitleBuffer::new(1 << 12), TitleBuffer::new(1 << 12)];
        let slots = vec![0u64; 8 * 4];
        let at = |i: u32| slots.as_ptr() as u64 + 32 * u64::from(i);
        // Each list's table: the title's view of `src` (waiting for a
        // sweep), then its output (made by a draw's table).
        for (k, dst) in dsts.iter().enumerate() {
            let slot = 2 * k as u32;
            let View::Shader(r, Some(words)) = buffer_view(false, &src, structured(16), (0, 16))
            else {
                unreachable!()
            };
            crate::descriptor::create_srv(
                std::ptr::null(),
                r as crate::object::This,
                &words,
                at(slot),
            );
            heap.mirror(slot, at(slot), false);
            heap.make_at(slot + 1, &buffer_view(true, dst, structured(16), (0, 16)));
        }
        let lists = [TitleList::new(), TitleList::new()];
        for (k, list) in lists.iter().enumerate() {
            let l = list.host();
            let picked = heap.picked_spans(at(0)).expect("a listing");
            PickedUse::default().defer(l, &picked, false);
            let base = 64 * k as u32;
            record_table_program(
                l,
                &heap,
                &pipeline,
                &cs,
                (&src, &dsts[k]),
                base,
                Reads::Step,
            );
            type Close = extern "system" fn(crate::object::This) -> crate::object::HRESULT;
            let close = crate::object::slot::<Close>(list.this(), 0x48);
            assert_eq!(close(list.this()), crate::object::S_OK);
        }
        let (one, two) = (lists[0].host(), lists[1].host());
        for value in [5000, 6000] {
            src.set_words(|i| value + i as u32);
            dsts.iter().for_each(|d| d.set_words(|_| 0));
            host.submit(&[one, two]);
            host.drain();
            // [2, 4] of 16-byte elements (dword 9) stored at [1, 8] (dword
            // 6); 10 added twice at [2, 0] (dword 8).
            for (k, dst) in dsts.iter().enumerate() {
                let out = dst.words();
                assert_eq!((out[6], out[8]), (value + 9, 20), "list {k}, {value}");
            }
        }
        RUN_TIME_VIEWS_TEST.set(None);
        // SAFETY: a plain device query.
        assert!(unsafe { host.device.GetDeviceRemovedReason() }.is_ok());
    }

    /// Codex review of 8612e27, P2 (a new bug): two dynamic slots waited
    /// for a raw view of one window; the first raw view back made it for
    /// the first slot, and `reap` stopped there, the pool full again,
    /// though the second slot could name it at no cost: it read zeros until
    /// another came back. It names it now, the pool full or not.
    #[test]
    fn slots_waiting_for_one_window_share_the_raw_view_made_for_it() {
        use crate::testgpu::TitleBuffer;
        let test =
            "descriptor::host::tests::slots_waiting_for_one_window_share_the_raw_view_made_for_it";
        let Some(gpu) = crate::testgpu::device(test) else {
            return;
        };
        let host = gpu.host();
        let heap = HostHeap::with_pool(8, 2).expect("a host view heap");
        let (a, b, c) = (
            TitleBuffer::new(1 << 12),
            TitleBuffer::new(1 << 12),
            TitleBuffer::new(1 << 12),
        );
        // Title slots 0 and 1 hold the pool's two raw views.
        heap.make_at(0, &buffer_view(false, &a, structured(16), (0, 8)));
        heap.make_at(1, &buffer_view(false, &b, structured(16), (0, 8)));
        assert_eq!(heap.pool_state(), (0, 0), "the pool is full");
        // Two dynamic slots read one window of `c`, as 16- and 32-byte
        // elements: no raw view for them.
        let first = heap.dynamic(2).expect("dynamic slots");
        heap.make_at(first, &buffer_view(false, &c, structured(16), (0, 16)));
        heap.make_at(first + 1, &buffer_view(false, &c, structured(32), (0, 8)));
        assert_eq!(heap.entry(first), StrideEntry::NONE);
        assert_eq!(heap.entry(first + 1), StrideEntry::NONE);
        // Slot 1's view goes: once the GPU is past what may read its raw
        // view, the first slot waiting makes `c`'s window with it, and the
        // second names it too, the pool full again.
        heap.make_null(1);
        completed_until(host, "never given one", || {
            heap.entry(first).raw_slot.is_some()
        });
        let (one, two) = (heap.entry(first), heap.entry(first + 1));
        assert_eq!(two.raw_slot, one.raw_slot, "the same raw view");
        assert_eq!((one.stride, one.count), (16, 16));
        assert_eq!((two.stride, two.count), (32, 8));
        assert_eq!(heap.pool_state(), (0, 0), "one pool slot for both");
        heap.release(first, 2);
        // SAFETY: a plain device query.
        assert!(unsafe { host.device.GetDeviceRemovedReason() }.is_ok());
    }

    /// What a pick refreshes of a host copy is its views' runs: ranges that
    /// overlap or lie less than a page apart merge, ranges farther apart
    /// stay apart (the span from the first view to the last refreshed what
    /// no view reads between them, unmapped memory among it). Runs merged
    /// as views come are the ones made afresh from the views, and a view
    /// that goes splits them again.
    #[test]
    fn views_merge_into_runs_closer_than_a_page() {
        const GAP: u64 = RUN_GAP;
        let mut runs = Runs::default();
        let list = |r: &mut Runs| r.runs().iter().map(|(&a, &b)| (a, b)).collect::<Vec<_>>();
        runs.add(0x10_0000, 0x10_0100);
        runs.add(0x10_0100 + GAP, 0x10_0200 + GAP);
        runs.add(0x40_0000, 0x40_0010);
        assert_eq!(
            list(&mut runs),
            [(0x10_0000, 0x10_0200 + GAP), (0x40_0000, 0x40_0010)],
            "a page apart: one run; far: two"
        );
        // One past the gap from both: its own run.
        runs.add(0x10_0201 + 2 * GAP, 0x10_0300 + 2 * GAP);
        assert_eq!(list(&mut runs).len(), 3);
        // Bridging them all.
        runs.add(0x10_0000, 0x40_0000);
        assert_eq!(list(&mut runs), [(0x10_0000, 0x40_0010)]);
        // The bridge goes: the runs of before.
        runs.remove(0x10_0000, 0x40_0000);
        assert_eq!(list(&mut runs).len(), 3);
        // The same range twice goes once: still there.
        runs.add(0x40_0000, 0x40_0010);
        runs.remove(0x40_0000, 0x40_0010);
        assert_eq!(list(&mut runs).len(), 3);
        // Merged as they come or made afresh, the same runs, whatever the
        // order (a fixed pseudo-random sequence of adds and removes).
        let mut seed = 0x2545_f491_4f6c_dd1du64;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        let (mut kept, mut live) = (Runs::default(), Vec::new());
        for _ in 0..2000 {
            if live.len() > 8 && next() % 3 == 0 {
                let (a, b) = live.swap_remove((next() % live.len() as u64) as usize);
                kept.remove(a, b);
            } else {
                let a = next() % (64 * GAP);
                let b = a + 1 + next() % (2 * GAP);
                kept.add(a, b);
                live.push((a, b));
            }
            let mut fresh = Runs::default();
            for &(a, b) in &live {
                fresh.add(a, b);
            }
            fresh.stale = true;
            assert_eq!(list(&mut kept), list(&mut fresh));
        }
    }

    /// Image-status §5.5: one raw view per slot ran out the pool about 30 s
    /// into the world (65,536 of 65,536, none retiring), and buffers read
    /// through the table read zeros from then on. Slots whose views read
    /// the same window name one raw view: a pool of 2 holds three slots of
    /// one window and one of another, and the shared one retires only once
    /// no slot names it.
    #[test]
    fn slots_viewing_the_same_window_share_one_raw_view() {
        use crate::testgpu::TitleBuffer;
        let _serial = crate::host::submissions::serial();
        let test = "descriptor::host::tests::slots_viewing_the_same_window_share_one_raw_view";
        let Some(gpu) = crate::testgpu::device(test) else {
            return;
        };
        let host = gpu.host();
        let heap = HostHeap::with_pool(8, 2).expect("a host view heap");
        let (a, b) = (TitleBuffer::new(1 << 12), TitleBuffer::new(1 << 12));
        let pool = heap.alias_base();
        for slot in 0..3 {
            heap.make_at(slot, &buffer_view(false, &a, structured(16), (0, 8)));
        }
        // The same bytes read as 32-byte elements: the same window.
        heap.make_at(3, &buffer_view(false, &a, structured(32), (0, 4)));
        for slot in 0..4 {
            assert_eq!(heap.entry(slot).raw_slot, Some(pool), "slot {slot}");
        }
        assert_eq!(heap.entry(3).stride, 32, "its own entry");
        assert_eq!(heap.pool_state(), (0, 1), "one raw view taken");
        heap.make_at(4, &buffer_view(false, &b, structured(16), (0, 8)));
        assert_eq!(heap.entry(4).raw_slot, Some(pool + 1));
        assert_eq!(heap.pool_state(), (0, 0));
        // Unordered access is another raw view: none left for it.
        heap.make_at(5, &buffer_view(true, &a, structured(16), (0, 8)));
        assert_eq!(heap.entry(5), StrideEntry::NONE, "the pool is full");
        // A submission in flight while they go, which may read them.
        let mut gate = crate::host::Gate::close(host);
        let running = host.new_list().expect("a host list");
        running.close();
        host.execute(&[&running]);
        for slot in 0..3 {
            heap.make_null(slot);
            assert_eq!(heap.pool_state(), (0, 0), "slot 3 still names it");
        }
        heap.make_null(3);
        assert_eq!(heap.pool_state(), (1, 0), "named by none: retiring");
        assert_eq!(heap.entry(4).raw_slot, Some(pool + 1), "the other stays");
        gate.open();
        completed_until(host, "never given back", || heap.pool_state() == (0, 1));
        // SAFETY: a plain device query.
        assert!(unsafe { host.device.GetDeviceRemovedReason() }.is_ok());
    }

    #[test]
    fn srv_range_resolves_buffer_srvs_and_skips_uav_and_cbv_views() {
        let test =
            "descriptor::host::tests::srv_range_resolves_buffer_srvs_and_skips_uav_and_cbv_views";
        let Some(gpu) = crate::testgpu::device(test) else {
            return;
        };
        let _ = gpu.host();
        let a = crate::testgpu::TitleBuffer::new(4096);
        let (copy, _, start, len) =
            srv_range(&buffer_view(false, &a, structured(16), (2, 8))).expect("buffer SRV range");
        assert_eq!(len, 8 * 16);
        assert_eq!(start, copy.offset(a.address()) + 2 * 16);
        assert!(srv_range(&buffer_view(true, &a, structured(16), (0, 8))).is_none());
        assert!(srv_range(&View::Constant(a.address(), 256)).is_none());
    }
}
