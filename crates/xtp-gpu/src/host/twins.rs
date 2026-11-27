//! Colour twins of depth textures in lists ([`crate::resource::Twin`]):
//! which side a list uses and writes, the copies between the two, and the
//! pass that draws a depth plane from its twin.
//!
//! Ported from fxceless's VIctory fork (commit 578d459: `TwinUse`,
//! `twin_use`, `record_twin_copy`, `record_plane_pass`, the prelude's
//! copies), onto XTP's lists.

use std::cell::Cell;
use std::collections::HashMap;
use std::sync::atomic::Ordering;
use std::sync::{Arc, LazyLock, Mutex, OnceLock, Weak};

use windows::Win32::Foundation::RECT;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::core::Interface;

use super::{HostList, host, transition};
use crate::resource::{HostTexture, Twin};

/// What a list does with a depth texture that has a colour twin. Lists are
/// recorded on many threads and run in submission order, so a list cannot
/// know which side an earlier list leaves behind: it notes which sides it
/// uses as it finds them, and its prelude copies one side into the other
/// at submission if that side is behind then. From its first use of a side
/// on, the list knows.
pub(super) struct TwinUse {
    /// The depth texture (the title resource's host copy) and its twin.
    depth: Arc<HostTexture>,
    twin: Arc<Twin>,
    /// The sides (depth texture, twin) the list uses as it finds them.
    needs: [bool; 2],
    /// Whether each side lacks what the other was written last, where the
    /// list is now; `None`: as the list found it. The twin here is
    /// `Twin::colour`: the whole twin, or the stencil plane's for a texture
    /// with one (its depth plane's is `plane_behind`).
    behind: [Option<bool>; 2],
    /// What the draw or dispatch being prepared writes: `Twin::colour`,
    /// the depth plane's twin.
    writing: [bool; 2],
    /// It also writes a twin through a view not named yet
    /// (`HostList::twin_view_written`): both, if it never is.
    unsure: bool,
    /// The depth plane lacks what its twin was written since the plane's
    /// last pass, which waits for the next use of the depth texture, draw,
    /// dispatch or query (`planes_now`), or for the list's close.
    plane_behind: bool,
    /// A pass drew the depth plane before the list's close: the close
    /// leaves the depth texture readable by shaders all the same, as when
    /// the pass waits for it (`twins_to_depth`).
    passed: bool,
}

/// The twins a list uses, by twin, and the viewports and scissor rectangles
/// the title set in it last: a plane's pass recorded between its commands
/// sets them back (`planes_now`).
#[derive(Default)]
pub(super) struct TwinUses {
    by_twin: HashMap<usize, TwinUse>,
    /// `None`: the title set none.
    viewports: Option<Vec<D3D12_VIEWPORT>>,
    scissors: Option<Vec<RECT>>,
    /// `XTP_PLANE_NOW` for this recording, read when first needed: one
    /// value for the whole list, even if an A/B turns it in the middle.
    now: Option<bool>,
    /// A pass was recorded between the title's commands, before a use of
    /// its depth texture or a query: the next draw or dispatch sets the
    /// title's state back (`planes_now`).
    ours: bool,
    /// Queries of the title's open in the list, which would count what a
    /// pass draws (`HostList::twins_query`): in one, planes are copied
    /// into instead (`HostList::copy_plane`).
    queries: u32,
    /// Probes of the passes recorded in the list, until it is submitted
    /// (`HostList::twins_submitted`).
    #[cfg(feature = "diagnostics")]
    probes: Vec<probe::Probe>,
}

impl TwinUses {
    /// The list is reset: it uses nothing, nothing is set in it.
    pub(super) fn clear(&mut self) {
        *self = TwinUses::default();
    }

    /// Whether depth planes are drawn from their twins between the title's
    /// commands in this list (`plane_now`).
    fn now(&mut self) -> bool {
        *self.now.get_or_insert_with(plane_now)
    }

    /// Whether the next draw or dispatch has something to do here, or the
    /// end of the one being prepared (`HostList::twins_due`).
    fn due(&self) -> bool {
        let planes = self.now != Some(false);
        self.ours
            || self.by_twin.values().any(|u| {
                u.unsure
                    || u.writing.contains(&true)
                    || planes
                        && (u.plane_behind || u.twin.plane.is_some() && u.behind[0] == Some(true))
            })
    }

    /// What is due before the next draw or dispatch: for each depth
    /// texture with a depth plane twin, the copy of the stencil plane's
    /// twin into it and the plane's pass, if they hold what it lacks.
    fn take_due(&mut self) -> Vec<Due> {
        let due: Vec<Due> = self
            .by_twin
            .values_mut()
            .filter(|u| u.twin.plane.is_some() && (u.plane_behind || u.behind[0] == Some(true)))
            .map(|u| {
                let copy = u.behind[0] == Some(true);
                if copy {
                    u.behind[0] = Some(false);
                }
                let pass = std::mem::take(&mut u.plane_behind);
                u.passed |= pass;
                Due {
                    depth: u.depth.clone(),
                    twin: u.twin.clone(),
                    copy,
                    pass,
                }
            })
            .collect();
        self.ours |= due.iter().any(|d| d.pass);
        due
    }
}

/// A depth texture's copy from its twin and its depth plane's pass, to
/// record (`TwinUses::take_due`).
struct Due {
    depth: Arc<HostTexture>,
    twin: Arc<Twin>,
    copy: bool,
    pass: bool,
}

thread_local! {
    /// `HostList::twin_target`: the next use of a depth texture on this
    /// thread is the depth target of the draw being prepared.
    static TARGET: Cell<bool> = const { Cell::new(false) };
}

/// `XTP_PLANE_NOW=0`: a depth plane is drawn from its twin only when its
/// list closes, as before `planes_now`; `ab` alternates the two within a
/// run (`xtp_core::ab`). Read once; its value for the frame being prepared.
fn plane_now() -> bool {
    static NOW: LazyLock<xtp_core::ab::Switch<bool>> = LazyLock::new(|| {
        let switch = xtp_core::ab::switch("XTP_PLANE_NOW", true);
        let mode = match (switch.alternates(), switch.get()) {
            (true, _) => "at their list's close in A/B phase A, before the next draw in phase B",
            (false, true) => "before the next draw or dispatch",
            (false, false) => "when their list closes",
        };
        tracing::info!("depth planes are drawn from their twins {mode} (XTP_PLANE_NOW)");
        switch
    });
    NOW.get()
}

/// A copy the prelude records: the depth texture, its twin, and whether
/// into the twin.
pub(super) type TwinSync = (Arc<HostTexture>, Arc<Twin>, bool);

const UNORDERED_ACCESS: u32 = 0x8;
const DEPTH_WRITE: u32 = 0x10;
const NON_PIXEL_SHADER_RESOURCE: u32 = 0x40;
const SHADER_RESOURCE: u32 = 0x40 | 0x80;
const PIXEL_SHADER_RESOURCE: u32 = 0x80;
const COPY_DEST: u32 = 0x400;
const COPY_SOURCE: u32 = 0x800;

impl HostList {
    /// Moves host texture `copy` to `wanted` (a host state) in this list,
    /// as [`HostList::transition`] does a title resource's host copy, and
    /// records the barrier: for colour twins (no title resource's copy)
    /// and the copies between a twin and its depth texture.
    pub(super) fn move_to(&self, copy: &Arc<HostTexture>, wanted: u32) {
        let wanted = super::legal_state(&copy.texture, wanted);
        let mut states = self.states.lock().unwrap_or_else(|e| e.into_inner());
        let before = match states.entry(Arc::as_ptr(copy) as usize) {
            std::collections::hash_map::Entry::Vacant(v) => {
                v.insert((wanted, wanted, copy.clone()));
                None
            }
            std::collections::hash_map::Entry::Occupied(mut o) => {
                let before = std::mem::replace(&mut o.get_mut().1, wanted);
                (before != wanted).then_some(before)
            }
        };
        drop(states);
        if let Some(before) = before {
            let barrier = transition(
                &copy.texture,
                D3D12_RESOURCE_STATES(before as i32),
                D3D12_RESOURCE_STATES(wanted as i32),
            );
            // SAFETY: an open list and a live texture.
            unsafe { self.list.ResourceBarrier(&[barrier]) };
        }
    }

    /// A use of title texture `resource` about to be recorded, if it is a
    /// depth texture with a colour twin (false if it is not): of the twin
    /// (`colour`: unordered access) or of the depth texture (depth target,
    /// clear, shader read, copy, resolve), writing it or only reading. The
    /// side used is first brought up to date: by a copy recorded here if
    /// this list wrote the other side since, by the list's prelude if the
    /// list uses the side as it finds it. The twin is left in the unordered
    /// access state; the caller moves the depth texture to the state its use
    /// needs, as for any texture. A twin still ahead when the list closes
    /// is copied then (`twins_to_depth`).
    ///
    /// A depth plane whose twin shaders wrote since its last pass is drawn
    /// here, before this use of its depth texture: a copy, resolve or clear
    /// of `crate::list`, which then reads or overwrites what the shaders
    /// wrote. Before, the pass waited for the next draw or dispatch: such a
    /// copy took the classes before the shaders', and the pass then drew
    /// over what a clear or copy had just written. Draws and dispatches
    /// draw the planes first (`planes_now`), so none is due when they get
    /// here, but for the depth target of a draw whose own shaders write the
    /// twin (`twin_target`): its pipeline is set, and the pass comes after
    /// it, before the next draw or dispatch.
    /// shortcut: every depth target bound for a draw counts as written
    /// (also when the draw only tests), and every unordered access view a
    /// shader names as written; views a shader picks from the heap by a
    /// computed index are not seen here. Inside a query of the title's, the
    /// plane is copied into rather than drawn (`copy_plane`): a draw would
    /// count in the query. Before, it waited for the query's end, and a
    /// clear or copy in the query met the plane as it was.
    pub(crate) fn twin_use(
        &self,
        resource: crate::object::This,
        colour: bool,
        write: bool,
    ) -> bool {
        let target = TARGET.replace(false);
        // Asked for every texture a draw reads: most have no twin.
        let Some(twin) = crate::resource::twin(resource) else {
            return false;
        };
        let Some(depth) = crate::resource::host_copy(resource) else {
            return false;
        };
        let side = usize::from(colour);
        let mut twins = self.twins.lock().unwrap_or_else(|e| e.into_inner());
        let now = twins.now();
        let in_query = twins.queries > 0;
        let u = twins
            .by_twin
            .entry(Arc::as_ptr(&twin) as usize)
            .or_insert_with(|| TwinUse {
                depth: depth.clone(),
                twin: twin.clone(),
                needs: [false; 2],
                behind: [None; 2],
                writing: [false; 2],
                unsure: false,
                plane_behind: false,
                passed: false,
            });
        let copy = match u.behind[side] {
            Some(behind) => behind,
            None => {
                u.needs[side] = true;
                false
            }
        };
        u.behind[side] = Some(false);
        let behind = !colour && u.plane_behind;
        let pass = behind && now && !target;
        if pass {
            u.plane_behind = false;
            u.passed = true;
        }
        match (write, colour) {
            // Shaders write the twin when the draw or dispatch runs: what
            // the same one reads of the depth texture is prepared before
            // (`twins_written`). Which twin, the view tells
            // (`twin_view_written`).
            (true, true) => {
                if std::mem::replace(&mut u.unsure, true) {
                    u.writing = [true; 2];
                }
            }
            (true, false) => u.behind[1] = Some(true),
            (false, _) => {}
        }
        twins.ours |= pass;
        drop(twins);
        if pass || write && colour {
            self.twins_due.store(true, Ordering::Relaxed);
        }
        #[cfg(feature = "diagnostics")]
        if twin.plane.is_some() {
            stale::used(colour, write, behind, behind && !pass);
        }
        if copy {
            self.twin_copy(&depth, &twin, colour);
        } else if colour {
            self.move_to(&twin.colour, UNORDERED_ACCESS);
        }
        if pass {
            self.plane_between(&depth, &twin, in_query);
        }
        // The depth plane's twin is written through unordered access too.
        if colour && let Some(plane) = &twin.plane {
            self.move_to(plane, UNORDERED_ACCESS);
        }
        true
    }

    /// `draw::State::prepare_for`, just before the use of the depth target
    /// of the draw being prepared (`twin_use`): its pipeline and tables are
    /// set, so no pass of ours may come before it.
    pub(crate) fn twin_target(&self) {
        TARGET.set(true);
    }

    /// After `twin_use` of title texture `resource` through unordered
    /// access view `view`, which writes it: the twin the view is of. For a
    /// depth texture with a stencil plane, the view is of the stencil
    /// plane's twin or of the depth plane's (`crate::resource::twin_view`);
    /// only the twin written is ahead of its plane. Before, both were, and a
    /// write of the stencil alone had the depth plane drawn from a twin
    /// that lacked what the list had cleared or copied into the plane.
    pub(crate) fn twin_view_written(
        &self,
        resource: crate::object::This,
        view: &crate::descriptor::View,
    ) {
        let crate::descriptor::View::Unordered(_, _, desc) = view else {
            return;
        };
        let Some(twin) = crate::resource::twin(resource) else {
            return;
        };
        let Some(plane) = &twin.plane else {
            return;
        };
        let desc = desc.as_ref().map(unordered_desc);
        let Some((texture, _)) = crate::resource::twin_view(resource, desc) else {
            return;
        };
        let to_plane = texture.as_raw() == plane.texture.as_raw();
        let mut twins = self.twins.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(u) = twins.by_twin.get_mut(&(Arc::as_ptr(&twin) as usize))
            && std::mem::take(&mut u.unsure)
        {
            u.writing[usize::from(to_plane)] = true;
        }
    }

    /// Records the copy of a depth texture into its colour twin (`colour`)
    /// or of the twin into the depth texture, with the moves to the copy
    /// states. The twin is left in the unordered access state.
    fn twin_copy(&self, depth: &Arc<HostTexture>, twin: &Twin, colour: bool) {
        let (from, to) = if colour {
            (depth, &twin.colour)
        } else {
            (&twin.colour, depth)
        };
        self.move_to(from, COPY_SOURCE);
        self.move_to(to, COPY_DEST);
        record_twin_copy(&self.list, &depth.texture, twin, colour);
        self.move_to(&twin.colour, UNORDERED_ACCESS);
    }

    /// Records the pass that draws the twin of the depth plane of `depth`
    /// into the plane, if `twin` has one, with the moves to the states the
    /// pass needs; the twin is left in the unordered access state. At the
    /// list's close, or between the title's commands, after which the next
    /// draw or dispatch sets the title's state again (`planes_now`).
    fn pass_plane(&self, depth: &Arc<HostTexture>, twin: &Twin) {
        let Some(plane) = &twin.plane else { return };
        self.move_to(depth, DEPTH_WRITE);
        self.move_to(plane, PIXEL_SHADER_RESOURCE);
        record_plane_pass(self, &depth.texture, plane);
        self.move_to(plane, UNORDERED_ACCESS);
    }

    /// `pass_plane` inside a query of the title's, which would count what
    /// the pass draws: the twin goes into the plane by a dispatch and
    /// copies, which no occlusion query counts (a pipeline statistics query
    /// counts the dispatch's threads). For each mip and slice, the dispatch
    /// writes each texel of the twin as the plane holds it into rows laid
    /// out as the host copies the plane, and the rows are copied into the
    /// plane, whole subresources. 16-bit steps go through the table of
    /// what the pass draws for each (`PlanePass::steps_table`), so the
    /// plane gets the bits the pass would have drawn. The depth texture is
    /// left in the copy destination state, the twin in unordered access;
    /// the list's descriptor heap, compute root signature and pipeline are
    /// ours. False if the host has no pipeline, table or buffer for it:
    /// nothing is recorded.
    /// shortcut: floats go into a 24-bit plane as WARP converts them
    /// ([`UNORM24`]); D3D lets another host round them differently by up to
    /// 0.6 of a step, where the pass has the host's own rounding. BTP 6's
    /// planes drawn from twins are 24-bit planes of 16-bit steps, which the
    /// table gives exactly on any host.
    fn copy_plane(&self, depth: &Arc<HostTexture>, twin: &Twin) -> bool {
        let Some(plane) = &twin.plane else {
            return true;
        };
        let Some(host) = host() else { return false };
        let pass = plane_pass();
        // SAFETY: live textures.
        let (of, steps) = unsafe { (depth.texture.GetDesc(), plane.texture.GetDesc()) };
        // An R16 typeless twin (53) holds steps, an R32 typeless one
        // floats; R24G8 depth (44 to 47) is 24-bit, R32G8X24 (19 to 22) a
        // float.
        let kind = match (steps.Format.0 == 53, of.Format.0) {
            (true, 44..=47) => PlaneCopy::Steps,
            (false, 44..=47) => PlaneCopy::Floats24,
            (false, 19..=22) => PlaneCopy::Floats32,
            _ => return false,
        };
        let table = match kind {
            PlaneCopy::Steps => match pass.steps_table() {
                Some(table) => Some(table),
                None => return false,
            },
            PlaneCopy::Floats24 | PlaneCopy::Floats32 => None,
        };
        // The depth plane's subresources come first, mip by mip in each
        // slice, as the views of the twin (`PlaneViews`).
        let planes = u32::from(of.MipLevels) * u32::from(of.DepthOrArraySize);
        let mut rows = vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); planes as usize];
        let mut bytes = 0;
        // SAFETY: a plain query writing `planes` footprints.
        unsafe {
            host.device.GetCopyableFootprints(
                &of,
                0,
                planes,
                0,
                Some(rows.as_mut_ptr()),
                None,
                None,
                Some(&mut bytes),
            );
        }
        let (Some((root, pipeline)), Some(views), Some(through)) = (
            pass.copier(kind),
            pass.views(plane),
            pass.rows(plane, bytes),
        ) else {
            return false;
        };
        self.keep(&views.heap);
        self.keep(&through);
        if let Some(table) = &table {
            self.keep(table);
        }
        self.move_to(plane, NON_PIXEL_SHADER_RESOURCE);
        self.move_to(depth, COPY_DEST);
        let l = &self.list;
        let (common, unordered, source) = (
            D3D12_RESOURCE_STATE_COMMON,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COPY_SOURCE,
        );
        // SAFETY: an open list; our heap, buffers and pipeline, kept by the
        // list (the pipeline lives as long as the process); each dispatch
        // writes inside its subresource's footprint, which the buffer was
        // made as big as; the buffer rests in the common state between
        // uses, the textures are in the states set above.
        unsafe {
            let base = through.GetGPUVirtualAddress();
            l.SetDescriptorHeaps(&[Some(views.heap.clone())]);
            l.SetComputeRootSignature(&root);
            l.SetPipelineState(&pipeline);
            if let Some(table) = &table {
                l.SetComputeRootShaderResourceView(3, table.GetGPUVirtualAddress());
            }
            l.ResourceBarrier(&[transition(&through, common, unordered)]);
            for (i, row) in rows.iter().enumerate() {
                let f = row.Footprint;
                l.SetComputeRootDescriptorTable(
                    0,
                    D3D12_GPU_DESCRIPTOR_HANDLE {
                        ptr: views.gpu + i as u64 * u64::from(views.step),
                    },
                );
                l.SetComputeRootUnorderedAccessView(1, base + row.Offset);
                let size = [f.RowPitch, f.Width, f.Height];
                l.SetComputeRoot32BitConstants(2, 3, size.as_ptr().cast(), 0);
                l.Dispatch(f.Width.div_ceil(8), f.Height.div_ceil(8), 1);
            }
            l.ResourceBarrier(&[transition(&through, unordered, source)]);
            for (i, row) in rows.iter().enumerate() {
                l.CopyTextureRegion(
                    &subresource_at(&depth.texture, i as u32),
                    0,
                    0,
                    0,
                    &placed_at(&through, row),
                    None,
                );
            }
            l.ResourceBarrier(&[transition(&through, source, common)]);
        }
        self.move_to(plane, UNORDERED_ACCESS);
        true
    }

    /// Records the depth plane of `depth` drawn from its twin between the
    /// title's commands: by the pass, or inside a query of the title's
    /// (`in_query`) by a dispatch and copies (`copy_plane`). If the host
    /// cannot copy it, the plane stays behind its twin and waits for the
    /// next draw or dispatch after the query, as before.
    fn plane_between(&self, depth: &Arc<HostTexture>, twin: &Arc<Twin>, in_query: bool) {
        let recorded = if in_query {
            self.copy_plane(depth, twin)
        } else {
            self.pass_plane(depth, twin);
            true
        };
        if !recorded {
            let mut twins = self.twins.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(u) = twins.by_twin.get_mut(&(Arc::as_ptr(twin) as usize)) {
                u.plane_behind = true;
            }
            self.twins_due.store(true, Ordering::Relaxed);
        }
        #[cfg(feature = "diagnostics")]
        if recorded {
            stale::pass(true);
            probe::record(self, depth, twin, true);
        }
    }

    /// Records `due` (`TwinUses::take_due`) between the title's commands,
    /// inside a query of the title's if `in_query` (`plane_between`).
    /// Each depth texture is left in the state the list had it in, or
    /// readable by shaders if the list had not used it yet (as at the
    /// close) or only through unordered access, which leaves a depth
    /// texture in the common state (`legal_state`), where no shader reads
    /// it.
    fn record_due(&self, due: &[Due], in_query: bool) {
        for d in due {
            let state = match self.state_now(&d.depth) {
                None | Some(0) => SHADER_RESOURCE,
                Some(state) => state,
            };
            if d.copy {
                self.twin_copy(&d.depth, &d.twin, false);
            }
            if d.pass {
                self.plane_between(&d.depth, &d.twin, in_query);
            }
            self.move_to(&d.depth, state);
        }
    }

    /// Before a draw or dispatch sets anything (`draw::State::bind`): the
    /// depth planes whose twins shaders of this list wrote since their last
    /// pass are drawn now, with the copy of the stencil plane, not when the
    /// list closes, so that what reads them next in the list finds them.
    /// The material resolve and the tiled lighting's classification write
    /// each pixel's material class into such a twin, which passes then read
    /// as depth; before, they saw the classes of the frame before.
    /// Done for whatever comes next (a pass more if it is another write of
    /// the twin, never a wrong one): which textures a draw reads is known
    /// only while its tables are made, after its pipeline is set.
    /// True if a pass was recorded since the title's last draw or dispatch
    /// (here, or before a copy, resolve or clear of a depth texture, or a
    /// query: `twin_use`, `twins_query`): it set a descriptor heap, a
    /// graphics root signature, a pipeline, a depth target, a viewport, a
    /// scissor and the topology of its own. The title's viewports and
    /// scissors are set back here (none if it set none, as the list
    /// started), the caller sets the rest (`draw::State::after_our_pass`).
    /// In a query of the title's, a pass would count in it: the planes are
    /// copied into instead (`copy_plane`), which sets a descriptor heap, a
    /// compute root signature and a pipeline of its own, set back the same
    /// way. Before, they waited for the query's end, and a draw in the
    /// query tested the classes of before.
    /// `XTP_PLANE_NOW=0` keeps the old way.
    ///
    /// A list that has nothing due takes no lock here.
    ///
    /// Ported from ZSlayerHQ/victory-world-runtime#8 (ef8bcaba2) by
    /// Geckomacabre. Their version leaves the depth texture in the depth
    /// write state and does not take up a plane whose copy the list already
    /// made.
    pub(crate) fn planes_now(&self) -> bool {
        if !self.twins_due.load(Ordering::Relaxed) {
            return false;
        }
        let mut twins = self.twins.lock().unwrap_or_else(|e| e.into_inner());
        let due = if twins.now() {
            twins.take_due()
        } else {
            Vec::new()
        };
        let in_query = twins.queries > 0;
        let ours = std::mem::take(&mut twins.ours);
        let title = ours.then(|| (twins.viewports.clone(), twins.scissors.clone()));
        self.twins_due.store(twins.due(), Ordering::Relaxed);
        drop(twins);
        // A plane the host cannot copy in a query is due again after this
        // (`plane_between`).
        self.record_due(&due, in_query);
        let Some((viewports, scissors)) = title else {
            return false;
        };
        // SAFETY: an open list.
        unsafe {
            self.list
                .RSSetViewports(viewports.as_deref().unwrap_or_default());
            self.list
                .RSSetScissorRects(scissors.as_deref().unwrap_or_default());
        }
        true
    }

    /// The state this list leaves `copy` in at this point, if it used it
    /// already.
    fn state_now(&self, copy: &Arc<HostTexture>) -> Option<u32> {
        let states = self.states.lock().unwrap_or_else(|e| e.into_inner());
        states.get(&(Arc::as_ptr(copy) as usize)).map(|s| s.1)
    }

    /// `draw::viewports`: the viewports the title sets, which `planes_now`
    /// sets back after its pass (noted whatever `XTP_PLANE_NOW` says: an
    /// A/B may turn it on in the middle of a list).
    pub(crate) fn note_viewports(&self, viewports: &[D3D12_VIEWPORT]) {
        let mut twins = self.twins.lock().unwrap_or_else(|e| e.into_inner());
        let noted = twins.viewports.get_or_insert_with(Vec::new);
        noted.clear();
        noted.extend_from_slice(viewports);
    }

    /// `draw::scissors`: the same for the scissor rectangles.
    pub(crate) fn note_scissors(&self, rects: &[RECT]) {
        let mut twins = self.twins.lock().unwrap_or_else(|e| e.into_inner());
        let noted = twins.scissors.get_or_insert_with(Vec::new);
        noted.clear();
        noted.extend_from_slice(rects);
    }

    /// `crate::list`'s BeginQuery (`open`) and EndQuery of the title's,
    /// of host query type `kind`. An occlusion query counts the samples a
    /// pass draws (its depth test passes them all), a pipeline statistics
    /// query its vertices and pixels: the planes due are drawn before a
    /// query opens, and those due inside one are copied into
    /// (`copy_plane`). Before round 2, a dispatch in an open occlusion
    /// query drew a plane in it, and an object the title had found hidden
    /// counted as visible; in round 2, they waited for its end, and what
    /// the title drew, cleared or copied in the query met the classes of
    /// before.
    pub(crate) fn twins_query(&self, kind: u32, open: bool) {
        const TIMESTAMP: u32 = 2;
        if kind == TIMESTAMP {
            return;
        }
        let mut twins = self.twins.lock().unwrap_or_else(|e| e.into_inner());
        if !open {
            twins.queries = twins.queries.saturating_sub(1);
            return;
        }
        let due = if twins.queries == 0 && self.twins_due.load(Ordering::Relaxed) && twins.now() {
            twins.take_due()
        } else {
            Vec::new()
        };
        twins.queries += 1;
        self.twins_due.store(twins.due(), Ordering::Relaxed);
        drop(twins);
        // Before the query opens: drawn.
        self.record_due(&due, false);
    }

    /// Before the list closes: the depth textures whose twins it wrote last
    /// are given what was written, and left readable by shaders, as are
    /// those whose depth plane it drew from its twin before (the pass may
    /// have left them where the title last used them). Shaders that pick a
    /// view from the heap by a computed index are not seen by `twin_use`
    /// and read the depth texture in lists that never name it.
    pub(super) fn twins_to_depth(&self) {
        let mut twins = self.twins.lock().unwrap_or_else(|e| e.into_inner());
        let written: Vec<_> = twins
            .by_twin
            .values_mut()
            .filter_map(|u| {
                let copy = u.behind[0] == Some(true);
                if copy {
                    u.behind[0] = Some(false);
                }
                let pass = std::mem::take(&mut u.plane_behind);
                (copy || pass || u.passed).then(|| Due {
                    depth: u.depth.clone(),
                    twin: u.twin.clone(),
                    copy,
                    pass,
                })
            })
            .collect();
        drop(twins);
        for d in &written {
            if d.copy {
                self.twin_copy(&d.depth, &d.twin, false);
            }
            // Nothing of the title's is recorded after this: the pass may
            // leave its own pipeline and target set.
            if d.pass {
                #[cfg(feature = "diagnostics")]
                stale::pass(false);
                self.pass_plane(&d.depth, &d.twin);
                #[cfg(feature = "diagnostics")]
                probe::record(self, &d.depth, &d.twin, false);
            }
            self.move_to(&d.depth, SHADER_RESOURCE);
        }
    }

    /// The draw or dispatch just prepared is recorded: the twins it writes
    /// now hold what their planes lack.
    pub(crate) fn twins_written(&self) {
        if !self.twins_due.load(Ordering::Relaxed) {
            return;
        }
        let mut twins = self.twins.lock().unwrap_or_else(|e| e.into_inner());
        for u in twins.by_twin.values_mut() {
            let wrote = if std::mem::take(&mut u.unsure) {
                [true; 2]
            } else {
                u.writing
            };
            u.writing = [false; 2];
            if wrote[0] {
                u.behind[0] = Some(true);
            }
            if wrote[1] && u.twin.plane.is_some() {
                u.plane_behind = true;
            }
        }
        self.twins_due.store(twins.due(), Ordering::Relaxed);
    }

    /// At submission: the copies the prelude records first, for the sides
    /// the list uses as it finds them and an earlier list left behind; the
    /// sides are then as this list leaves them.
    pub(super) fn twin_syncs(&self) -> Vec<TwinSync> {
        let mut syncs = Vec::new();
        let twins = self.twins.lock().unwrap_or_else(|e| e.into_inner());
        // The list runs again: its probes wait for this run (diagnostics).
        #[cfg(feature = "diagnostics")]
        probe::submitting(&twins.probes);
        for u in twins.by_twin.values() {
            let real = u.twin.behind.load(Ordering::Relaxed);
            let mut behind = [real & 1 != 0, real & 2 != 0];
            for side in [0, 1] {
                if u.needs[side] && behind[side] {
                    syncs.push((u.depth.clone(), u.twin.clone(), side == 1));
                    behind = [false; 2];
                }
            }
            let left = [0, 1].map(|side| u.behind[side].unwrap_or(behind[side]));
            let left = u32::from(left[0]) | (u32::from(left[1]) << 1);
            u.twin.behind.store(left, Ordering::Relaxed);
        }
        syncs
    }

    /// `draw::State::bind`, before anything of a draw or dispatch is
    /// recorded: the pipeline (`name`: its programs) the uses of twins that
    /// follow on this thread are counted for (diagnostics, [`stale`]).
    #[cfg(feature = "diagnostics")]
    pub(crate) fn twin_reader(&self, name: &str) {
        stale::reader(name);
    }

    /// `Host::run`: the list was submitted, in a submission whose last
    /// command signals `fence`: the probes of its passes are read once the
    /// GPU has reached it (diagnostics, [`probe`]). The list keeps them
    /// until it is reset: submitted again, it writes their buffers again,
    /// and they are read once that run is over.
    #[cfg(feature = "diagnostics")]
    pub(crate) fn twins_submitted(&self, fence: u64) {
        let twins = self.twins.lock().unwrap_or_else(|e| e.into_inner());
        if !twins.probes.is_empty() {
            probe::submitted(&twins.probes, fence);
        }
    }
}

/// A title's unordered access view description, in the host's layout, as
/// `crate::descriptor::host` makes the view.
fn unordered_desc(words: &[u32; 10]) -> D3D12_UNORDERED_ACCESS_VIEW_DESC {
    let mut words = *words;
    words[0] = crate::resource::pc_view_format(&words, true);
    // SAFETY: 40 bytes in the host's layout.
    unsafe {
        std::ptr::from_ref(&words)
            .cast::<D3D12_UNORDERED_ACCESS_VIEW_DESC>()
            .read_unaligned()
    }
}

/// Diagnostics: how often a depth plane is used while its twin holds what
/// it lacks, for the frame line (`log`). A use of the depth texture of a
/// depth plane's twin that shaders of the same list wrote since the plane
/// was last drawn is *late* (`TwinUse::plane_behind`); the pass is then
/// recorded before it (in a query, the copy: `HostList::copy_plane`),
/// unless it is a draw's depth target.
/// That use and those after it until the pass find the plane *stale*.
/// Uses are counted for the pipeline last bound on the thread
/// (`HostList::twin_reader`); the copies, resolves and clears of
/// `crate::list` count for that pipeline too.
#[cfg(feature = "diagnostics")]
pub(super) mod stale {
    use std::cell::RefCell;
    use std::collections::HashMap;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicU64, Ordering};

    thread_local! {
        /// The pipeline this thread binds (`reader`).
        static READER: RefCell<String> = const { RefCell::new(String::new()) };
    }

    /// Since the start: late uses; uses of a stale plane read only, and as
    /// a depth target or written (copies into it, clears); the plane's
    /// passes between the title's commands and at a list's close; frames.
    static LATE: AtomicU64 = AtomicU64::new(0);
    static STALE_READS: AtomicU64 = AtomicU64::new(0);
    static STALE_TARGETS: AtomicU64 = AtomicU64::new(0);
    static PASSES_NOW: AtomicU64 = AtomicU64::new(0);
    static PASSES_AT_CLOSE: AtomicU64 = AtomicU64::new(0);
    static FRAMES: AtomicU64 = AtomicU64::new(0);

    /// By pipeline, since the start: writes of the plane's twin, and uses
    /// of the depth texture with the plane up to date and stale, read only
    /// and as a target ([`Uses`]).
    static BY_READER: Mutex<Option<HashMap<String, Uses>>> = Mutex::new(None);

    /// Twin writes, fresh reads, stale reads, fresh target uses, stale
    /// target uses.
    type Uses = [u64; 5];

    /// The totals at the previous line, to give each line its frames.
    static LAST: Mutex<[u64; 6]> = Mutex::new([0; 6]);

    pub(super) fn reader(name: &str) {
        READER.with(|r| {
            let mut r = r.borrow_mut();
            r.clear();
            r.push_str(name);
        });
    }

    /// A use of a depth texture whose depth plane has a twin: of the twin
    /// (`colour`) or the depth texture, written or read only; `late`: the
    /// plane waited for its pass; `stale`: it still does for this use.
    pub(super) fn used(colour: bool, write: bool, late: bool, stale: bool) {
        if late {
            LATE.fetch_add(1, Ordering::Relaxed);
        }
        if stale {
            let total = if write { &STALE_TARGETS } else { &STALE_READS };
            total.fetch_add(1, Ordering::Relaxed);
        }
        let at = match (colour, write, stale) {
            (true, _, _) => 0,
            (false, false, false) => 1,
            (false, false, true) => 2,
            (false, true, false) => 3,
            (false, true, true) => 4,
        };
        let name = READER.with(|r| r.borrow().clone());
        let mut by = BY_READER.lock().unwrap_or_else(|e| e.into_inner());
        by.get_or_insert_with(HashMap::new).entry(name).or_default()[at] += 1;
    }

    /// A pass that draws a depth plane from its twin, between the title's
    /// commands (`now`) or at a list's close.
    pub(super) fn pass(now: bool) {
        let total = if now { &PASSES_NOW } else { &PASSES_AT_CLOSE };
        total.fetch_add(1, Ordering::Relaxed);
    }

    /// Frames presented since the previous call (each second).
    pub(crate) fn frames(frames: f64) {
        FRAMES.fetch_add(frames as u64, Ordering::Relaxed);
    }

    /// The line: per frame since the previous line, the totals since the
    /// start, and the pipelines using planes, most uses first. The probes
    /// are read after `BY_READER` is let go: lists recording meanwhile
    /// count their uses without waiting for them.
    pub(crate) fn log() {
        let now = [
            &LATE,
            &STALE_READS,
            &STALE_TARGETS,
            &PASSES_NOW,
            &PASSES_AT_CLOSE,
            &FRAMES,
        ]
        .map(|c| c.load(Ordering::Relaxed));
        let mut last = LAST.lock().unwrap_or_else(|e| e.into_inner());
        let frames = now[5].saturating_sub(last[5]).max(1) as f64;
        let [late, reads, targets, passes_now, passes_at_close, _] =
            std::array::from_fn(|i| now[i].saturating_sub(last[i]) as f64 / frames);
        *last = now;
        drop(last);
        if now[0] + now[3] + now[4] == 0 {
            return;
        }
        let readers: Vec<String> = {
            let by = BY_READER.lock().unwrap_or_else(|e| e.into_inner());
            let mut readers: Vec<_> = by.iter().flatten().collect();
            readers
                .sort_by_key(|(name, uses)| (std::cmp::Reverse(uses.iter().sum::<u64>()), *name));
            readers
                .iter()
                .take(16)
                .map(|(name, [w, fr, sr, ft, st])| {
                    let name = name.replace('"', "");
                    let name = if name.is_empty() { "-" } else { name.as_str() };
                    format!("{name}: {w} twin writes, reads {fr} fresh {sr} stale, targets {ft} fresh {st} stale")
                })
                .collect()
        };
        super::probe::log();
        tracing::info!(
            "depth planes from twins, per frame: {late:.2} late uses, uses of a stale plane \
             {reads:.2} read and {targets:.2} as target, passes {passes_now:.2} between the \
             title's commands and {passes_at_close:.2} at a list's close; since the start ({} \
             frames): {} late, {} stale reads, {} stale target uses, {} passes between, {} at \
             close; by pipeline: {readers:?}",
            now[5],
            now[0],
            now[1],
            now[2],
            now[3],
            now[4],
        );
    }
}

/// Diagnostics, `XTP_PLANE_PROBE=1`: in the frames the capture lists
/// (`XTP_CAPTURE_FRAMES`, `XTP_CAPTURE_WORLD`), each depth plane's pass is
/// followed by copies of the twin it drew (the material classes) and of the
/// plane it drew into, read back once the GPU has run the list's submission
/// (`log`): the classes on both sides of the pass and the texels where they
/// differ. A plane keeps what its last pass drew until the next one (draws
/// test it without writing it), so this is what the lighting's "equal"
/// rectangles test against.
#[cfg(feature = "diagnostics")]
mod probe {
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::{Arc, Mutex, OnceLock};

    use windows::Win32::Graphics::Direct3D12::*;
    use windows::Win32::Graphics::Dxgi::Common::*;

    use super::{COPY_SOURCE, HostList, HostTexture, Twin, UNORDERED_ACCESS, host};

    /// A probe's fence while a submission of its list is being prepared,
    /// and before the first: never reached.
    const PENDING: u64 = u64::MAX;

    /// A pass's read-backs: subresource 0 of the twin and of the plane,
    /// each a buffer and its row pitch. Its list and [`PROBES`] share it.
    #[derive(Clone)]
    pub(super) struct Probe {
        frame: u64,
        /// Between the title's commands, or at the close.
        now: bool,
        /// The twin holds 16-bit steps (else floats); the plane is 24-bit
        /// (else a float).
        steps: bool,
        unorm24: bool,
        width: u32,
        height: u32,
        twin: (ID3D12Resource, u32),
        plane: (ID3D12Resource, u32),
        /// The fence value the last submission of the probe's list signals
        /// once run (`submitted`), [`PENDING`] while the next is prepared
        /// (`submitting`).
        fence: Arc<AtomicU64>,
    }

    // SAFETY: D3D12 objects are free-threaded.
    unsafe impl Send for Probe {}

    /// The probes of submitted lists, until their fence is reached.
    static PROBES: Mutex<Vec<Probe>> = Mutex::new(Vec::new());

    /// The frame being recorded, if probed.
    fn wanted() -> Option<u64> {
        static ON: OnceLock<bool> = OnceLock::new();
        if !*ON.get_or_init(|| std::env::var("XTP_PLANE_PROBE").is_ok_and(|v| v.trim() == "1")) {
            return None;
        }
        let frame = crate::queue::presented() + 1;
        crate::capture::listed(frame).then_some(frame)
    }

    /// Records into `list` the copy of subresource 0 of `texture` (in the
    /// copy source state) into a read-back buffer of ours: the buffer and
    /// its row pitch.
    fn copy(list: &HostList, texture: &ID3D12Resource) -> Option<(ID3D12Resource, u32)> {
        let host = host()?;
        // SAFETY: a live texture.
        let desc = unsafe { texture.GetDesc() };
        let mut footprint = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
        let mut bytes = 0;
        // SAFETY: a plain query for one subresource.
        unsafe {
            host.device.GetCopyableFootprints(
                &desc,
                0,
                1,
                0,
                Some(&mut footprint),
                None,
                None,
                Some(&mut bytes),
            );
        }
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_READBACK,
            ..Default::default()
        };
        let buffer = D3D12_RESOURCE_DESC {
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
        let mut readback: Option<ID3D12Resource> = None;
        // SAFETY: plain creation of a read-back buffer.
        unsafe {
            host.device
                .CreateCommittedResource(
                    &heap,
                    D3D12_HEAP_FLAG_NONE,
                    &buffer,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    None,
                    &mut readback,
                )
                .ok()?;
        }
        let readback = readback?;
        let to = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: a borrowed pointer the location never releases.
            pResource: unsafe { std::mem::transmute_copy(&readback) },
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: footprint,
            },
        };
        let from = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: as above.
            pResource: unsafe { std::mem::transmute_copy(texture) },
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                SubresourceIndex: 0,
            },
        };
        // SAFETY: an open list; both live, the texture in the copy source
        // state, the buffer as big as the footprint.
        unsafe { list.list.CopyTextureRegion(&to, 0, 0, 0, &from, None) };
        list.keep(&readback);
        Some((readback, footprint.Footprint.RowPitch))
    }

    /// After the pass that drew `twin`'s depth plane into `depth` (left in
    /// the depth write state, the twin in unordered access): the copies of
    /// both, if the frame is probed. The depth texture is left in the copy
    /// source state, the twin in unordered access.
    pub(super) fn record(list: &HostList, depth: &Arc<HostTexture>, twin: &Twin, now: bool) {
        if let Some(frame) = wanted() {
            record_in(list, depth, twin, now, frame);
        }
    }

    /// [`record`] for frame `frame`: the probe waits in the list until it
    /// is submitted.
    pub(super) fn record_in(
        list: &HostList,
        depth: &Arc<HostTexture>,
        twin: &Twin,
        now: bool,
        frame: u64,
    ) {
        let Some(plane) = &twin.plane else { return };
        list.move_to(depth, COPY_SOURCE);
        list.move_to(plane, COPY_SOURCE);
        let copies = (copy(list, &plane.texture), copy(list, &depth.texture));
        list.move_to(plane, UNORDERED_ACCESS);
        let (Some(twin_copy), Some(plane_copy)) = copies else {
            return;
        };
        // SAFETY: live textures.
        let (of, steps) = unsafe { (depth.texture.GetDesc(), plane.texture.GetDesc()) };
        list.twins
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .probes
            .push(Probe {
                frame,
                now,
                steps: steps.Format.0 == 53,
                unorm24: of.Format.0 == 44,
                width: of.Width as u32,
                height: of.Height,
                twin: twin_copy,
                plane: plane_copy,
                fence: Arc::new(AtomicU64::new(PENDING)),
            });
    }

    /// `HostList::twin_syncs`, before a submission of the probes' list
    /// runs: they wait for it, wherever they are.
    pub(super) fn submitting(probes: &[Probe]) {
        for p in probes {
            p.fence.store(PENDING, Ordering::SeqCst);
        }
    }

    /// `HostList::twins_submitted`: `probes` are read once the GPU has
    /// reached `fence`, the last submission of their list. Before, a probe
    /// was read 3 s after it was recorded, whether its list had run or
    /// not: a list kept longer (or never submitted) gave two untouched
    /// buffers, "no texel differs". In round 2, a probe was read once the
    /// first submission of its list had run, even if the list was
    /// submitted again meanwhile, which wrote the same buffers again.
    pub(super) fn submitted(probes: &[Probe], fence: u64) {
        let mut all = PROBES.lock().unwrap_or_else(|e| e.into_inner());
        for p in probes {
            p.fence.store(fence, Ordering::SeqCst);
            if !all.iter().any(|q| Arc::ptr_eq(&q.fence, &p.fence)) {
                all.push(p.clone());
            }
        }
    }

    /// The probes whose lists' last submissions the GPU has run, once
    /// `completed` is the fence value it has reached.
    fn ready(completed: u64) -> Vec<Probe> {
        ready_of(completed, None)
    }

    /// [`ready`], only those of `frame` if given (tests, which run side by
    /// side, take only theirs).
    pub(super) fn ready_of(completed: u64, frame: Option<u64>) -> Vec<Probe> {
        let mut probes = PROBES.lock().unwrap_or_else(|e| e.into_inner());
        let (ready, waiting) = std::mem::take(&mut *probes).into_iter().partition(|p| {
            reached(p.fence.load(Ordering::SeqCst), completed) && frame.is_none_or(|f| f == p.frame)
        });
        *probes = waiting;
        ready
    }

    /// Whether a probe whose fence is `fence` has run, once the GPU has
    /// reached `completed`.
    fn reached(fence: u64, completed: u64) -> bool {
        fence != PENDING && fence <= completed
    }

    /// The bytes of a read-back buffer.
    fn bytes(buffer: &ID3D12Resource) -> Option<Vec<u8>> {
        // SAFETY: a live buffer.
        let len = unsafe { buffer.GetDesc() }.Width as usize;
        let mut data = std::ptr::null_mut();
        // SAFETY: a read-back buffer whose copy has run (`ready`); its
        // mapping holds `len` bytes.
        unsafe {
            buffer.Map(0, None, Some(&mut data)).ok()?;
            let bytes = std::slice::from_raw_parts(data.cast::<u8>(), len).to_vec();
            buffer.Unmap(0, None);
            Some(bytes)
        }
    }

    /// The step the lighting and the pass multiply classes by.
    fn step() -> f64 {
        f64::from(f32::from_bits(0x3780_0080))
    }

    /// Whether a depth texel holds what a twin's float `bits` give it: the
    /// same float, or in a 24-bit plane (`unorm24`, its depth `z`) the
    /// float's nearest 24-bit value, give or take the one step D3D12 lets a
    /// conversion round either way. Before, a 24-bit plane was compared bit
    /// for bit with the float: every texel whose class is not a multiple of
    /// a 24-bit step differed.
    pub(super) fn same_float(z: f64, unorm24: bool, bits: u32) -> bool {
        let float = f64::from(f32::from_bits(bits));
        if unorm24 {
            let max = f64::from(0xff_ffffu32);
            ((float.clamp(0.0, 1.0) * max).round() - (z * max).round()).abs() <= 1.0
        } else {
            (z as f32).to_bits() == bits
        }
    }

    /// What a probe found: the classes in the twin and in the plane (class:
    /// texels, most first), the texels where they differ, all texels.
    pub(super) struct Found {
        pub(super) classes: Vec<(u32, u64)>,
        pub(super) drawn: Vec<(u32, u64)>,
        pub(super) differ: u64,
        pub(super) texels: u64,
    }

    /// Reads `p`'s buffers; `None` if they cannot be mapped.
    pub(super) fn found(p: &Probe) -> Option<Found> {
        let (Some(twin), Some(plane)) = (bytes(&p.twin.0), bytes(&p.plane.0)) else {
            return None;
        };
        let (tp, pp) = (p.twin.1 as usize, p.plane.1 as usize);
        let mut classes: HashMap<u32, u64> = HashMap::new();
        let mut drawn: HashMap<u32, u64> = HashMap::new();
        let (mut differ, mut texels) = (0u64, 0u64);
        let word = |b: &[u8], at: usize| {
            b.get(at..at + 4)
                .map_or(0, |w| u32::from_le_bytes([w[0], w[1], w[2], w[3]]))
        };
        let step = step();
        for y in 0..p.height as usize {
            for x in 0..p.width as usize {
                let z = if p.unorm24 {
                    f64::from(word(&plane, y * pp + x * 4) & 0xff_ffff) / f64::from(0xff_ffffu32)
                } else {
                    f64::from(f32::from_bits(word(&plane, y * pp + x * 4)))
                };
                let (class, same) = if p.steps {
                    let at = y * tp + x * 2;
                    let c = twin
                        .get(at..at + 2)
                        .map_or(0, |w| u16::from_le_bytes([w[0], w[1]]));
                    let est = (z / step).round();
                    (u32::from(c), est == f64::from(c))
                } else {
                    let bits = word(&twin, y * tp + x * 4);
                    (
                        (f64::from(f32::from_bits(bits)) / step).round() as u32,
                        same_float(z, p.unorm24, bits),
                    )
                };
                *classes.entry(class).or_default() += 1;
                *drawn.entry((z / step).round() as u32).or_default() += 1;
                differ += u64::from(!same);
                texels += 1;
            }
        }
        let top = |h: HashMap<u32, u64>| {
            let mut h: Vec<_> = h.into_iter().collect();
            h.sort_by_key(|&(c, n)| (std::cmp::Reverse(n), c));
            h.truncate(8);
            h
        };
        Some(Found {
            classes: top(classes),
            drawn: top(drawn),
            differ,
            texels,
        })
    }

    /// The frame line's: one line per probe whose list has run. A probe
    /// whose list is submitted again while it is read is skipped: that run
    /// may have written what was read, and the probe is read again once it
    /// is over (`submitted`).
    pub(super) fn log() {
        let Some(host) = host() else { return };
        let completed = host.completed();
        for p in ready(completed) {
            let fence = p.fence.load(Ordering::SeqCst);
            if !reached(fence, completed) {
                continue;
            }
            let found = found(&p);
            std::sync::atomic::fence(Ordering::SeqCst);
            if p.fence.load(Ordering::SeqCst) != fence {
                continue;
            }
            let Some(found) = found else { continue };
            tracing::info!(
                frame = p.frame,
                when = if p.now {
                    "between the title's commands"
                } else {
                    "at the list's close"
                },
                size = format_args!("{}x{}", p.width, p.height),
                differ = found.differ,
                texels = found.texels,
                "depth plane probe: classes in the twin (class: texels) {:?}, in the plane {:?}",
                found.classes,
                found.drawn,
            );
        }
    }
}

/// The prelude's part for `syncs`: the barriers moving each texture a copy
/// uses from its real state into the copy's (`to_copy`), those moving it
/// back where the list does not use it (`back`, before the list's own
/// barriers), and the state each is in once copied, by host copy.
pub(super) fn sync_states(
    syncs: &[TwinSync],
    used: &super::States,
) -> (
    Vec<D3D12_RESOURCE_BARRIER>,
    Vec<D3D12_RESOURCE_BARRIER>,
    std::collections::HashMap<usize, u32>,
) {
    let host_state = |state: u32| D3D12_RESOURCE_STATES(state as i32);
    let mut to_copy = Vec::new();
    let mut back = Vec::new();
    let mut copying = std::collections::HashMap::new();
    for (depth, twin, colour) in syncs {
        let (from, to) = if *colour {
            (depth, &twin.colour)
        } else {
            (&twin.colour, depth)
        };
        for (texture, state) in [(from, COPY_SOURCE), (to, COPY_DEST)] {
            let key = Arc::as_ptr(texture) as usize;
            if copying.contains_key(&key) {
                continue;
            }
            let state = super::legal_state(&texture.texture, state);
            let real = texture.state.load(Ordering::Relaxed);
            copying.insert(key, state);
            if real != state {
                to_copy.push(transition(
                    &texture.texture,
                    host_state(real),
                    host_state(state),
                ));
                if !used.contains_key(&key) {
                    back.push(transition(
                        &texture.texture,
                        host_state(state),
                        host_state(real),
                    ));
                }
            }
        }
    }
    (to_copy, back, copying)
}

/// The pass that draws the twin of a depth plane into the plane: one
/// triangle over each mip of each slice, its pixel shader writing the
/// twin's texel as the depth, with the depth test "always" and no stencil
/// (the stencil plane is copied through a buffer and not touched here).
///
/// A 16-bit twin holds whole steps of 1/65535, and the depth written is
/// the step times the float nearest 1/65535 (0x37800080). That is how BTP
/// 6's tiled lighting computes the depth of the rectangles it then draws
/// with the test "equal", one class of materials a draw (its vertex
/// shaders multiply the class number by that constant), so the pass and
/// the draw give the host the same float, and the host turns the same
/// float into the same 24 bits.
///
/// The twin's view is volatile data, as the copy's (`copy_source`): the
/// pass's root signature stays set while shaders write the twin again and
/// the title dispatches, which binds only its compute side.
const PLANE_CORNERS: &str = r#"
float4 corner(uint i : SV_VertexID) : SV_Position {
    return float4(float(i & 1) * 4.0 - 1.0, float(i >> 1) * 4.0 - 1.0, 0.0, 1.0);
}
"#;
const PLANE_STEPS: &str = r#"
#define RS "DescriptorTable(SRV(t0, flags = DATA_VOLATILE), visibility = SHADER_VISIBILITY_PIXEL)"
Texture2DArray<uint> twin : register(t0);
[RootSignature(RS)]
float plane(float4 p : SV_Position) : SV_Depth {
    return float(twin.Load(int4(int2(p.xy), 0, 0))) * asfloat(0x37800080);
}
"#;
const PLANE_FLOATS: &str = r#"
#define RS "DescriptorTable(SRV(t0, flags = DATA_VOLATILE), visibility = SHADER_VISIBILITY_PIXEL)"
Texture2DArray<float> twin : register(t0);
[RootSignature(RS)]
float plane(float4 p : SV_Position) : SV_Depth {
    return twin.Load(int4(int2(p.xy), 0, 0));
}
"#;

/// The table of what the pass draws into a 24-bit plane for each 16-bit
/// step (`PlanePass::steps_table`): texel `y * 256 + x` of a 256×256 plane
/// gets step `y * 256 + x`, computed as `PLANE_STEPS` computes it, so the
/// host turns the same float into the same 24 bits.
const PLANE_TABLE: &str = r#"
#define RS "RootFlags(0)"
[RootSignature(RS)]
float plane(float4 p : SV_Position) : SV_Depth {
    return float(uint(p.y) * 256 + uint(p.x)) * asfloat(0x37800080);
}
"#;

/// A float as a 24-bit depth (`PlaneCopy::Floats24`): clamped to 0..1
/// (NaN 0), times 2^24 - 1 in 32-bit floats, rounded to nearest even, as
/// WARP writes depth. D3D's rule (times 2^24 - 1, plus a half, truncated)
/// computed exactly gave one step more than WARP on 992 of 65,536 floats,
/// those whose product is within half a float step of a half: WARP rounds
/// the product to a float first, and an exact half to even.
const UNORM24: &str = r#"
uint unorm24(float f) {
    return uint(round(saturate(f) * 16777215.0));
}
"#;

/// How a copy in a query (`HostList::copy_plane`) gives a depth plane its
/// twin's texels.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum PlaneCopy {
    /// 16-bit steps into a 24-bit plane: each looked up in the table of
    /// what the pass draws for it (`PlanePass::steps_table`).
    Steps,
    /// Floats into a 24-bit plane: converted as WARP converts depth
    /// ([`UNORM24`]).
    Floats24,
    /// Floats into a 32-bit float plane: as they are, from 0 to 1, as the
    /// pass's viewport clamps the depth it writes.
    Floats32,
}

/// The compute shader of a copy of kind `kind`: each texel of one mip and
/// slice of the twin (`t0`), as its plane holds it, into rows `pitch`
/// bytes apart (`u0`, 4 bytes a texel, the layout of a copy of the plane);
/// for steps, through the table (`t1`). The twin's view is volatile data:
/// the copy's root signature stays set while shaders write the twin again
/// and the title draws, which binds only its graphics side (as static
/// data, the debug layer's message 1003 at the title's next draw).
fn copy_source(kind: PlaneCopy) -> String {
    let (table, texel, depth) = match kind {
        PlaneCopy::Steps => (", SRV(t1)", "uint", "table.Load(texel * 4) & 0xffffff"),
        PlaneCopy::Floats24 => ("", "float", "unorm24(texel)"),
        PlaneCopy::Floats32 => ("", "float", "asuint(saturate(texel))"),
    };
    format!(
        r#"
#define RS "DescriptorTable(SRV(t0, flags = DATA_VOLATILE)), UAV(u0), RootConstants(num32BitConstants = 3, b0){table}"
Texture2DArray<{texel}> twin : register(t0);
ByteAddressBuffer table : register(t1);
RWByteAddressBuffer rows : register(u0);
cbuffer Rows : register(b0) {{ uint pitch; uint width; uint height; }};
{UNORM24}
[RootSignature(RS)]
[numthreads(8, 8, 1)]
void copy(uint3 id : SV_DispatchThreadID) {{
    if (id.x < width && id.y < height) {{
        {texel} texel = twin.Load(int4(id.xy, 0, 0));
        rows.Store(id.y * pitch + id.x * 4, {depth});
    }}
}}
"#
    )
}

/// What the pass draws with: per kind of twin (16-bit steps or floats)
/// and depth format the root signature and pipeline, made when first
/// needed (`None`: the host refused), and the views of the twins it reads.
/// In a query of the title's, what its copies use (`HostList::copy_plane`).
struct PlanePass {
    made: Mutex<Vec<(bool, i32, Option<PlanePipeline>)>>,
    /// The copies' root signatures and pipelines by kind, made when first
    /// needed (`None`: the host refused).
    copiers: Mutex<Vec<(PlaneCopy, Option<PlanePipeline>)>>,
    /// By depth plane twin (its address): the twin, weakly, and the buffer
    /// its copies go through, as big as its depth texture's depth plane
    /// laid out for a copy. Lists recording at once may share it: each
    /// fills it and copies from it in turn, in the order they run.
    rows: Mutex<HashMap<usize, (Weak<HostTexture>, ID3D12Resource)>>,
    /// The table of 16-bit steps' depths in a 24-bit plane and the list of
    /// ours that makes it (`steps_table`), made when first needed (`None`:
    /// the host refused).
    table: OnceLock<Option<(ID3D12Resource, HostList)>>,
    /// By depth plane twin (its address): the twin, weakly, and the views
    /// of its mips and slices (`PlaneViews`). Made once and never written
    /// again, so a view outlives every list that reads it, each keeping
    /// its heap. Before, one ring of 256 views served every list: 256
    /// passes more, recorded before a list ran, wrote another twin's view
    /// into the slot it read. A twin gone takes its views with it when the
    /// next twin's are made.
    views: Mutex<HashMap<usize, (Weak<HostTexture>, PlaneViews)>>,
}

/// The views a depth plane twin's passes read, in a heap of ours: view
/// `slice * mips + mip` of the twin is at `gpu + index * step`.
#[derive(Clone)]
struct PlaneViews {
    heap: ID3D12DescriptorHeap,
    gpu: u64,
    step: u32,
}

/// The pass's root signature and pipeline for one kind of twin and one
/// depth format.
type PlanePipeline = (ID3D12RootSignature, ID3D12PipelineState);

// SAFETY: D3D12 objects are free-threaded.
unsafe impl Send for PlanePass {}
// SAFETY: as above.
unsafe impl Sync for PlanePass {}

fn plane_pass() -> &'static PlanePass {
    static PASS: LazyLock<PlanePass> = LazyLock::new(|| PlanePass {
        made: Mutex::default(),
        copiers: Mutex::default(),
        rows: Mutex::default(),
        table: OnceLock::new(),
        views: Mutex::default(),
    });
    &PASS
}

/// Compiles function `entry` of `source` for `target` with the system's
/// compiler.
fn compile(
    source: &str,
    entry: windows::core::PCSTR,
    target: windows::core::PCSTR,
) -> windows::core::Result<Vec<u8>> {
    use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
    use windows::Win32::Graphics::Direct3D::ID3DBlob;
    let mut code: Option<ID3DBlob> = None;
    let mut errors: Option<ID3DBlob> = None;
    // SAFETY: the source and names live through the call; out pointers ours.
    let compiled = unsafe {
        D3DCompile(
            source.as_ptr().cast(),
            source.len(),
            windows::core::s!("depth plane"),
            None,
            None,
            entry,
            target,
            0,
            0,
            &mut code,
            Some(&mut errors),
        )
    };
    if let Err(e) = compiled {
        if let Some(errors) = errors {
            // SAFETY: the blob's own bytes.
            let text = unsafe {
                std::slice::from_raw_parts(
                    errors.GetBufferPointer().cast::<u8>(),
                    errors.GetBufferSize(),
                )
            };
            tracing::warn!("the depth plane pass: {}", String::from_utf8_lossy(text));
        }
        return Err(e);
    }
    let code = code.ok_or_else(windows::core::Error::empty)?;
    // SAFETY: the blob's own bytes.
    let bytes = unsafe {
        std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize())
    };
    Ok(bytes.to_vec())
}

impl PlanePass {
    /// The root signature and pipeline for a twin of 16-bit steps (`steps`)
    /// or of floats and the depth format `format`.
    fn pipeline(&self, steps: bool, format: DXGI_FORMAT) -> Option<PlanePipeline> {
        let mut made = self.made.lock().unwrap_or_else(|e| e.into_inner());
        if let Some((_, _, pipeline)) = made.iter().find(|m| (m.0, m.1) == (steps, format.0)) {
            return pipeline.clone();
        }
        let pipeline = match Self::make(steps, format) {
            Ok(pipeline) => Some(pipeline),
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    steps,
                    format = format.0,
                    "no pipeline to draw a depth plane from its twin: such planes stay as they are"
                );
                None
            }
        };
        made.push((steps, format.0, pipeline.clone()));
        pipeline
    }

    fn make(steps: bool, format: DXGI_FORMAT) -> windows::core::Result<PlanePipeline> {
        Self::graphics(if steps { PLANE_STEPS } else { PLANE_FLOATS }, format)
    }

    /// A pipeline drawing one triangle over the target with pixel shader
    /// `plane` of `source` (its root signature inside) as the depth, into
    /// a depth target of `format`.
    fn graphics(source: &str, format: DXGI_FORMAT) -> windows::core::Result<PlanePipeline> {
        use windows::core::s;
        let device = &host().ok_or_else(windows::core::Error::empty)?.device;
        let vs = compile(PLANE_CORNERS, s!("corner"), s!("vs_5_1"))?;
        let ps = compile(source, s!("plane"), s!("ps_5_1"))?;
        let code = |bytes: &[u8]| D3D12_SHADER_BYTECODE {
            pShaderBytecode: bytes.as_ptr().cast(),
            BytecodeLength: bytes.len(),
        };
        // SAFETY: the root signature is inside the pixel shader; the
        // description points at live data and never releases its pointer.
        unsafe {
            let root: ID3D12RootSignature = device.CreateRootSignature(0, &ps[..])?;
            let desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&root),
                VS: code(&vs[..]),
                PS: code(&ps[..]),
                SampleMask: u32::MAX,
                RasterizerState: D3D12_RASTERIZER_DESC {
                    FillMode: D3D12_FILL_MODE_SOLID,
                    CullMode: D3D12_CULL_MODE_NONE,
                    DepthClipEnable: true.into(),
                    ..Default::default()
                },
                // Every texel is written; stencil stays off.
                DepthStencilState: D3D12_DEPTH_STENCIL_DESC {
                    DepthEnable: true.into(),
                    DepthWriteMask: D3D12_DEPTH_WRITE_MASK_ALL,
                    DepthFunc: D3D12_COMPARISON_FUNC_ALWAYS,
                    ..Default::default()
                },
                PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
                DSVFormat: format,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                ..Default::default()
            };
            let pipeline: ID3D12PipelineState = device.CreateGraphicsPipelineState(&desc)?;
            Ok((root, pipeline))
        }
    }

    /// The root signature and pipeline of a copy of kind `kind`
    /// (`HostList::copy_plane`).
    fn copier(&self, kind: PlaneCopy) -> Option<PlanePipeline> {
        let mut copiers = self.copiers.lock().unwrap_or_else(|e| e.into_inner());
        if let Some((_, pipeline)) = copiers.iter().find(|c| c.0 == kind) {
            return pipeline.clone();
        }
        let pipeline = match Self::compute(&copy_source(kind)) {
            Ok(pipeline) => Some(pipeline),
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    ?kind,
                    "no pipeline to copy a depth plane from its twin in a query: such planes \
                     wait for the query's end"
                );
                None
            }
        };
        copiers.push((kind, pipeline.clone()));
        pipeline
    }

    /// A compute pipeline of function `copy` of `source` (its root
    /// signature inside).
    fn compute(source: &str) -> windows::core::Result<PlanePipeline> {
        use windows::core::s;
        let device = &host().ok_or_else(windows::core::Error::empty)?.device;
        let cs = compile(source, s!("copy"), s!("cs_5_1"))?;
        // SAFETY: the root signature is inside the shader; the description
        // points at live data and never releases its pointer.
        unsafe {
            let root: ID3D12RootSignature = device.CreateRootSignature(0, &cs[..])?;
            let desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&root),
                CS: D3D12_SHADER_BYTECODE {
                    pShaderBytecode: cs.as_ptr().cast(),
                    BytecodeLength: cs.len(),
                },
                ..Default::default()
            };
            let pipeline: ID3D12PipelineState = device.CreateComputePipelineState(&desc)?;
            Ok((root, pipeline))
        }
    }

    /// The buffer the copies of depth plane twin `plane` go through,
    /// `bytes` long (`rows`), made on its first copy.
    fn rows(&self, plane: &Arc<HostTexture>, bytes: u64) -> Option<ID3D12Resource> {
        let key = Arc::as_ptr(plane) as usize;
        let mut rows = self.rows.lock().unwrap_or_else(|e| e.into_inner());
        if let Some((_, made)) = rows.get(&key) {
            return Some(made.clone());
        }
        rows.retain(|_, (twin, _)| twin.strong_count() > 0);
        let made = match buffer(bytes, true) {
            Ok(made) => made,
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    bytes,
                    "no buffer to copy a depth plane from its twin in a query: it waits for \
                     the query's end"
                );
                return None;
            }
        };
        rows.insert(key, (Arc::downgrade(plane), made.clone()));
        Some(made)
    }

    /// The table of what the pass draws into a 24-bit plane for each
    /// 16-bit step: word `step` of the buffer holds, in its low 24 bits,
    /// the depth the host writes for the step's float. Made on the first
    /// copy of steps, by a list of ours submitted then: every list that
    /// reads it is submitted after it, on the one host queue. Kept, with
    /// that list, as long as the process (dropping the list would wait for
    /// the GPU).
    fn steps_table(&self) -> Option<ID3D12Resource> {
        self.table
            .get_or_init(|| match Self::make_table() {
                Ok(made) => Some(made),
                Err(e) => {
                    tracing::warn!(
                        error = %e,
                        "no table of depths for 16-bit steps: in a query, such planes wait \
                         for its end"
                    );
                    None
                }
            })
            .as_ref()
            .map(|(table, _)| table.clone())
    }

    fn make_table() -> windows::core::Result<(ID3D12Resource, HostList)> {
        let host = host().ok_or_else(windows::core::Error::empty)?;
        let device = &host.device;
        let (root, pipeline) = Self::graphics(PLANE_TABLE, DXGI_FORMAT_D24_UNORM_S8_UINT)?;
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: 256,
            Height: 256,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: DXGI_FORMAT_R24G8_TYPELESS,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Flags: D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL,
            ..Default::default()
        };
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_DEFAULT,
            ..Default::default()
        };
        let mut texture: Option<ID3D12Resource> = None;
        let mut footprint = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
        let mut bytes = 0;
        // SAFETY: plain creation and a plain query for one subresource; the
        // out pointers are ours.
        unsafe {
            device.CreateCommittedResource(
                &heap,
                D3D12_HEAP_FLAG_NONE,
                &desc,
                D3D12_RESOURCE_STATE_DEPTH_WRITE,
                None,
                &mut texture,
            )?;
            device.GetCopyableFootprints(
                &desc,
                0,
                1,
                0,
                Some(&mut footprint),
                None,
                None,
                Some(&mut bytes),
            );
        }
        let texture = texture.ok_or_else(windows::core::Error::empty)?;
        // Step `y * 256 + x` at byte `(y * 256 + x) * 4`: rows of 1024 bytes.
        if footprint.Footprint.RowPitch != 1024 {
            return Err(windows::core::Error::new(
                windows::Win32::Foundation::E_UNEXPECTED,
                format!("rows {} bytes apart", footprint.Footprint.RowPitch),
            ));
        }
        let table = buffer(bytes, false)?;
        crate::host::name(&table, || "depths of 16-bit steps".to_string());
        let list = host.new_list()?;
        let target = D3D12_DEPTH_STENCIL_VIEW_DESC {
            Format: DXGI_FORMAT_D24_UNORM_S8_UINT,
            ViewDimension: D3D12_DSV_DIMENSION_TEXTURE2D,
            ..Default::default()
        };
        let target = host.dsv(&texture, Some(&target));
        let l = &list.list;
        // SAFETY: an open list of ours; live objects, kept by the list or
        // the process; the buffer as big as the footprint, promoted from
        // the common state by the copy.
        unsafe {
            l.SetGraphicsRootSignature(&root);
            l.SetPipelineState(&pipeline);
            l.IASetPrimitiveTopology(
                windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST,
            );
            l.OMSetRenderTargets(0, None, false, Some(&target));
            l.RSSetViewports(&[D3D12_VIEWPORT {
                Width: 256.0,
                Height: 256.0,
                MaxDepth: 1.0,
                ..Default::default()
            }]);
            l.RSSetScissorRects(&[RECT {
                left: 0,
                top: 0,
                right: 256,
                bottom: 256,
            }]);
            l.DrawInstanced(3, 1, 0, 0);
            l.ResourceBarrier(&[transition(
                &texture,
                D3D12_RESOURCE_STATE_DEPTH_WRITE,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            )]);
            l.CopyTextureRegion(
                &placed_at(&table, &footprint),
                0,
                0,
                0,
                &subresource_at(&texture, 0),
                None,
            );
        }
        list.keep(&texture);
        list.keep(&root);
        list.keep(&pipeline);
        list.close();
        host.execute(&[&list]);
        Ok((table, list))
    }
}

/// A buffer of ours in the card's memory, `bytes` long, in the common
/// state; written by shaders if `unordered`.
fn buffer(bytes: u64, unordered: bool) -> windows::core::Result<ID3D12Resource> {
    let device = &host().ok_or_else(windows::core::Error::empty)?.device;
    let heap = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_DEFAULT,
        ..Default::default()
    };
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: bytes.max(1),
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: if unordered {
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS
        } else {
            D3D12_RESOURCE_FLAG_NONE
        },
        ..Default::default()
    };
    let mut buffer: Option<ID3D12Resource> = None;
    // SAFETY: plain creation; the out pointer is ours.
    unsafe {
        device.CreateCommittedResource(
            &heap,
            D3D12_HEAP_FLAG_NONE,
            &desc,
            D3D12_RESOURCE_STATE_COMMON,
            None,
            &mut buffer,
        )?;
    }
    buffer.ok_or_else(windows::core::Error::empty)
}

/// Copy location: subresource `index` of `texture`.
fn subresource_at(texture: &ID3D12Resource, index: u32) -> D3D12_TEXTURE_COPY_LOCATION {
    D3D12_TEXTURE_COPY_LOCATION {
        // SAFETY: a borrowed pointer the location never releases.
        pResource: unsafe { std::mem::transmute_copy(texture) },
        Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
            SubresourceIndex: index,
        },
    }
}

/// Copy location: `footprint` in `buffer`.
fn placed_at(
    buffer: &ID3D12Resource,
    footprint: &D3D12_PLACED_SUBRESOURCE_FOOTPRINT,
) -> D3D12_TEXTURE_COPY_LOCATION {
    D3D12_TEXTURE_COPY_LOCATION {
        // SAFETY: a borrowed pointer the location never releases.
        pResource: unsafe { std::mem::transmute_copy(buffer) },
        Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
            PlacedFootprint: *footprint,
        },
    }
}

impl PlanePass {
    /// The views of depth plane twin `plane` (`PlaneViews`), made on its
    /// first pass.
    fn views(&self, plane: &Arc<HostTexture>) -> Option<PlaneViews> {
        let key = Arc::as_ptr(plane) as usize;
        let mut views = self.views.lock().unwrap_or_else(|e| e.into_inner());
        if let Some((_, made)) = views.get(&key) {
            return Some(made.clone());
        }
        views.retain(|_, (twin, _)| twin.strong_count() > 0);
        let made = match Self::make_views(&plane.texture) {
            Ok(made) => made,
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    "no views of a depth plane twin: its plane stays as it is"
                );
                return None;
            }
        };
        views.insert(key, (Arc::downgrade(plane), made.clone()));
        Some(made)
    }

    fn make_views(plane: &ID3D12Resource) -> windows::core::Result<PlaneViews> {
        let device = &host().ok_or_else(windows::core::Error::empty)?.device;
        // SAFETY: a live texture.
        let d = unsafe { plane.GetDesc() };
        let (mips, slices) = (u32::from(d.MipLevels), u32::from(d.DepthOrArraySize));
        // An R16 typeless twin (53) holds steps, read as R16_UINT (57); an
        // R32 typeless one floats, read as R32_FLOAT (41).
        let format = DXGI_FORMAT(if d.Format.0 == 53 { 57 } else { 41 });
        let kind = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        // SAFETY: plain heap creation and queries; views of a live texture
        // into slots of the new heap, which nothing reads yet.
        unsafe {
            let heap: ID3D12DescriptorHeap =
                device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: kind,
                    NumDescriptors: mips * slices,
                    Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                    NodeMask: 0,
                })?;
            crate::host::name(&heap, || "depth plane twin views".to_string());
            let step = device.GetDescriptorHandleIncrementSize(kind);
            let cpu = heap.GetCPUDescriptorHandleForHeapStart().ptr;
            for slice in 0..slices {
                for mip in 0..mips {
                    let view = D3D12_SHADER_RESOURCE_VIEW_DESC {
                        Format: format,
                        ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2DARRAY,
                        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                            Texture2DArray: D3D12_TEX2D_ARRAY_SRV {
                                MostDetailedMip: mip,
                                MipLevels: 1,
                                FirstArraySlice: slice,
                                ArraySize: 1,
                                ..Default::default()
                            },
                        },
                    };
                    let at = D3D12_CPU_DESCRIPTOR_HANDLE {
                        ptr: cpu + ((slice * mips + mip) * step) as usize,
                    };
                    device.CreateShaderResourceView(plane, Some(&view), at);
                }
            }
            Ok(PlaneViews {
                gpu: heap.GetGPUDescriptorHandleForHeapStart().ptr,
                heap,
                step,
            })
        }
    }
}

/// Records into `list` the pass that draws `plane`, the twin of the depth
/// plane of `depth`, into that plane: `depth` in the depth-write state,
/// `plane` readable by pixel shaders. The pass sets the list's descriptor
/// heap, graphics root signature, pipeline, depth target, viewport,
/// scissor and topology and puts none of them back (at a list's close, or
/// between the title's commands, after which `HostList::planes_now` and its
/// caller set the title's again). The list keeps the heap of the views it
/// reads.
pub(super) fn record_plane_pass(list: &HostList, depth: &ID3D12Resource, plane: &Arc<HostTexture>) {
    let Some(host) = host() else { return };
    let pass = plane_pass();
    // SAFETY: live resources.
    let (of, twin) = unsafe { (depth.GetDesc(), plane.texture.GetDesc()) };
    // An R16 typeless twin (53) holds steps, an R32 typeless one floats.
    // The depth view names the depth-stencil format of the texture's
    // family (44: 45, 19: 20).
    let steps = twin.Format.0 == 53;
    let format = DXGI_FORMAT(match of.Format.0 {
        44 => 45,
        19 => 20,
        typed => typed,
    });
    let Some((root, pipeline)) = pass.pipeline(steps, format) else {
        return;
    };
    let Some(views) = pass.views(plane) else {
        return;
    };
    list.keep(&views.heap);
    let mips = u32::from(of.MipLevels);
    let l = &list.list;
    // SAFETY: an open list; our heap, kept by the list; the pass's own
    // objects live as long as the process.
    unsafe {
        l.SetDescriptorHeaps(&[Some(views.heap.clone())]);
        l.SetGraphicsRootSignature(&root);
        l.SetPipelineState(&pipeline);
        l.IASetPrimitiveTopology(
            windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST,
        );
    }
    for slice in 0..u32::from(of.DepthOrArraySize) {
        for mip in 0..mips {
            let target = D3D12_DEPTH_STENCIL_VIEW_DESC {
                Format: format,
                ViewDimension: D3D12_DSV_DIMENSION_TEXTURE2DARRAY,
                Anonymous: D3D12_DEPTH_STENCIL_VIEW_DESC_0 {
                    Texture2DArray: D3D12_TEX2D_ARRAY_DSV {
                        MipSlice: mip,
                        FirstArraySlice: slice,
                        ArraySize: 1,
                    },
                },
                ..Default::default()
            };
            let target = host.dsv(depth, Some(&target));
            let (width, height) = ((of.Width >> mip).max(1) as u32, (of.Height >> mip).max(1));
            let gpu = D3D12_GPU_DESCRIPTOR_HANDLE {
                ptr: views.gpu + u64::from((slice * mips + mip) * views.step),
            };
            // SAFETY: an open list; a view of our heap, set above; live
            // resources.
            unsafe {
                l.SetGraphicsRootDescriptorTable(0, gpu);
                l.OMSetRenderTargets(0, None, false, Some(&target));
                l.RSSetViewports(&[D3D12_VIEWPORT {
                    TopLeftX: 0.0,
                    TopLeftY: 0.0,
                    Width: width as f32,
                    Height: height as f32,
                    MinDepth: 0.0,
                    MaxDepth: 1.0,
                }]);
                l.RSSetScissorRects(&[RECT {
                    left: 0,
                    top: 0,
                    right: width as i32,
                    bottom: height as i32,
                }]);
                l.DrawInstanced(3, 1, 0, 0);
            }
        }
    }
}

/// Records into `list` the copy of a depth texture into its colour twin
/// (`colour`) or of the twin into the depth texture, both already in their
/// copy states: whole textures of one size in one format family, which the
/// host copies as they are; or, for the twin of a stencil plane, that plane
/// alone, through the twin's buffer.
pub(super) fn record_twin_copy(
    list: &ID3D12GraphicsCommandList,
    depth: &ID3D12Resource,
    twin: &Twin,
    colour: bool,
) {
    let (from, to) = if colour {
        (depth, &twin.colour.texture)
    } else {
        (&twin.colour.texture, depth)
    };
    // shortcut: the depth plane's twin is not given the plane (that is a
    // pass the other way: the plane read by a shader, the twin as its
    // target). The twin then lacks what draws and clears wrote to the
    // plane since shaders last wrote the twin; in BTP 6 nothing writes that
    // plane but the twin's pass (its draws against it only test).
    let Some(through) = &twin.through else {
        // SAFETY: an open list and two live textures.
        unsafe { list.CopyResource(to, from) };
        return;
    };
    // A stencil plane and its twin: each mip and slice of the plane goes
    // into the buffer, laid out as the host lays the plane out for copies,
    // and from the buffer into the other texture, whole subresources both
    // times. In the depth texture the stencil plane's subresources follow
    // the depth plane's; the twin has the one plane.
    let Some(host) = host() else { return };
    // SAFETY: a live texture.
    let of = unsafe { depth.GetDesc() };
    let planes = u32::from(of.MipLevels) * u32::from(of.DepthOrArraySize);
    let mut rows = vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); planes as usize];
    // SAFETY: a plain query writing `planes` footprints.
    unsafe {
        host.device.GetCopyableFootprints(
            &of,
            planes,
            planes,
            0,
            Some(rows.as_mut_ptr()),
            None,
            None,
            None,
        );
    }
    let (first_from, first_to) = if colour { (planes, 0) } else { (0, planes) };
    let subresource = |texture: &ID3D12Resource, index: u32| D3D12_TEXTURE_COPY_LOCATION {
        // SAFETY: a borrowed pointer the location never releases.
        pResource: unsafe { std::mem::transmute_copy(texture) },
        Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
            SubresourceIndex: index,
        },
    };
    let placed = |row: &D3D12_PLACED_SUBRESOURCE_FOOTPRINT| D3D12_TEXTURE_COPY_LOCATION {
        // SAFETY: as above.
        pResource: unsafe { std::mem::transmute_copy(through) },
        Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
            PlacedFootprint: *row,
        },
    };
    let (common, source, dest) = (
        D3D12_RESOURCE_STATE_COMMON,
        D3D12_RESOURCE_STATE_COPY_SOURCE,
        D3D12_RESOURCE_STATE_COPY_DEST,
    );
    // SAFETY: an open list and live resources; the buffer was made as big
    // as these footprints, and rests in the common state between copies.
    unsafe {
        list.ResourceBarrier(&[transition(through, common, dest)]);
        for (i, row) in rows.iter().enumerate() {
            let index = first_from + i as u32;
            list.CopyTextureRegion(&placed(row), 0, 0, 0, &subresource(from, index), None);
        }
        list.ResourceBarrier(&[transition(through, dest, source)]);
        for (i, row) in rows.iter().enumerate() {
            let index = first_to + i as u32;
            list.CopyTextureRegion(&subresource(to, index), 0, 0, 0, &placed(row), None);
        }
        list.ResourceBarrier(&[transition(through, source, common)]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::object::This;

    /// A title depth texture of `format` with unordered access asked.
    fn depth(width: u64, height: u32, format: u32) -> This {
        let desc = crate::resource::tests_support::depth_unordered(width, height, format);
        crate::resource::tests_support::committed(&desc)
    }

    fn release(r: This) {
        crate::object::slot::<extern "system" fn(This) -> u32>(r, 0x10)(r);
    }

    /// Records the copy of `bytes` (rows laid out as subresource `index`'s
    /// footprint) into that subresource of `texture`, in the copy
    /// destination state.
    fn fill(list: &HostList, texture: &ID3D12Resource, index: u32, bytes: &[u8]) {
        let host = host().unwrap();
        // SAFETY: a live texture.
        let d = unsafe { texture.GetDesc() };
        let mut footprint = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
        // SAFETY: a plain query for one subresource.
        unsafe {
            host.device.GetCopyableFootprints(
                &d,
                index,
                1,
                0,
                Some(&mut footprint),
                None,
                None,
                None,
            );
        }
        let (upload, data) = host.upload_buffer(4096).unwrap();
        // SAFETY: the buffer's mapping holds 4096 bytes.
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), data, bytes.len().min(4096)) };
        footprint.Offset = 0;
        let to = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: a borrowed pointer the location never releases.
            pResource: unsafe { std::mem::transmute_copy(texture) },
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                SubresourceIndex: index,
            },
        };
        let from = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: as above.
            pResource: unsafe { std::mem::transmute_copy(&upload) },
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: footprint,
            },
        };
        // SAFETY: an open list and live resources.
        unsafe { list.list.CopyTextureRegion(&to, 0, 0, 0, &from, None) };
        list.keep(&upload);
    }

    /// Subresource `index` of `texture` (in `state`), rows 256 bytes apart
    /// (whatever the pitch of its copies: a depth plane's may be wider).
    fn read(texture: &ID3D12Resource, state: u32, index: u32) -> Vec<u8> {
        let host = host().unwrap();
        // SAFETY: a live texture.
        let d = unsafe { texture.GetDesc() };
        let mut footprint = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
        // SAFETY: a plain query for one subresource.
        unsafe {
            host.device.GetCopyableFootprints(
                &d,
                index,
                1,
                0,
                Some(&mut footprint),
                None,
                None,
                None,
            );
        }
        footprint.Offset = 0;
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_READBACK,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: 4096,
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
        let mut readback: Option<ID3D12Resource> = None;
        // SAFETY: plain creation of a readback buffer.
        unsafe {
            host.device
                .CreateCommittedResource(
                    &heap,
                    D3D12_HEAP_FLAG_NONE,
                    &desc,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    None,
                    &mut readback,
                )
                .unwrap();
        }
        let readback = readback.unwrap();
        let list = host.new_list().unwrap();
        let to = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: a borrowed pointer the location never releases.
            pResource: unsafe { std::mem::transmute_copy(&readback) },
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: footprint,
            },
        };
        let from = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: as above.
            pResource: unsafe { std::mem::transmute_copy(texture) },
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                SubresourceIndex: index,
            },
        };
        let state = D3D12_RESOURCE_STATES(super::super::legal_state(texture, state) as i32);
        let source = D3D12_RESOURCE_STATE_COPY_SOURCE;
        // SAFETY: an open list and live resources.
        unsafe {
            if state != source {
                list.list
                    .ResourceBarrier(&[transition(texture, state, source)]);
            }
            list.list.CopyTextureRegion(&to, 0, 0, 0, &from, None);
            if state != source {
                list.list
                    .ResourceBarrier(&[transition(texture, source, state)]);
            }
        }
        list.close();
        host.execute(&[&list]);
        host.drain();
        let mut mapped = std::ptr::null_mut();
        // SAFETY: the copy ran (drained); the buffer is 4096 bytes.
        unsafe {
            readback.Map(0, None, Some(&mut mapped)).unwrap();
            let bytes = std::slice::from_raw_parts(mapped.cast::<u8>(), 4096);
            let pitch = footprint.Footprint.RowPitch as usize;
            let mut rows = vec![0u8; 1024];
            for row in 0..(4096 / pitch).min(4) {
                rows[row * 256..][..pitch.min(256)]
                    .copy_from_slice(&bytes[row * pitch..][..pitch.min(256)]);
            }
            readback.Unmap(0, None);
            rows
        }
    }

    /// A 16-bit depth texture the title also writes through unordered
    /// access: what is put into its twin is in the depth texture once the
    /// list reads the depth texture after.
    #[test]
    fn a_depth_texture_reads_what_its_twin_was_written() {
        let test = "host::twins::tests::a_depth_texture_reads_what_its_twin_was_written";
        let Some(gpu) = crate::testgpu::device(test) else {
            return;
        };
        let host = gpu.host();
        const D16_UNORM: u32 = 55;
        let depth = depth(4, 4, D16_UNORM);
        // Made with the first unordered access view of the texture.
        assert!(crate::resource::twin(depth).is_none());
        assert!(crate::resource::twin_view(depth, None).is_some());
        let twin = crate::resource::twin(depth).expect("a twin");
        // SAFETY: live resources.
        let (d, t) = unsafe {
            (
                crate::resource::host_resource(depth).unwrap().GetDesc(),
                twin.colour.texture.GetDesc(),
            )
        };
        // Both R16 typeless: one a depth texture, one with unordered access.
        assert_eq!((d.Format.0, d.Flags.0), (53, 2));
        assert_eq!((t.Format.0, t.Flags.0), (53, 4));
        let mut pattern = vec![0u8; 1024];
        for row in 0..4 {
            for b in 0..8 {
                pattern[row * 256 + b] = (1 + row * 8 + b) as u8;
            }
        }
        let list = host.new_list().unwrap();
        // A shader writes the twin: here, a copy into it.
        assert!(list.twin_use(depth, true, true));
        list.move_to(&twin.colour, COPY_DEST);
        fill(&list, &twin.colour.texture, 0, &pattern);
        list.move_to(&twin.colour, UNORDERED_ACCESS);
        list.twins_written();
        // Then a draw tests against the depth texture.
        assert!(list.twin_use(depth, false, false));
        crate::list::ensure_state(&list, depth, DEPTH_WRITE);
        list.close();
        host.execute(&[&list]);
        host.drain();
        let copy = crate::resource::host_copy(depth).unwrap();
        let read = read(&copy.texture, copy.state.load(Ordering::Relaxed), 0);
        for row in 0..4 {
            let at = row * 256;
            assert_eq!(read[at..at + 8], pattern[at..at + 8], "row {row}");
        }
        release(depth);
    }

    /// Writes into the twin of the depth plane of `depth`, as a shader of
    /// the list would, class `left` in the two left columns and `right`
    /// elsewhere (steps of 1/65535, 4×4).
    fn write_classes(list: &HostList, depth: This, left: u16, right: u16) {
        assert!(list.twin_use(depth, true, true));
        let plane = crate::resource::twin(depth)
            .and_then(|t| t.plane.clone())
            .expect("a depth plane twin");
        let mut steps = vec![0u8; 1024];
        for row in 0..4 {
            for x in 0..4 {
                let class = if x < 2 { left } else { right };
                steps[row * 256 + x * 2..][..2].copy_from_slice(&class.to_le_bytes());
            }
        }
        list.move_to(&plane, COPY_DEST);
        fill(list, &plane.texture, 0, &steps);
        list.move_to(&plane, UNORDERED_ACCESS);
        list.twins_written();
    }

    /// A title pipeline that draws the whole screen at the depth of class
    /// `class` (the step times the float nearest 1/65535, as the lighting's
    /// vertices do), tested "equal" without writes against a console D16S8
    /// depth (D24S8 on the host), in white into an RGBA8 target.
    fn equal_pipeline(class: u16) -> This {
        use xtp_shader::HwStage;
        use xtp_shader::emit::{HostShader, rect_gs::probe};
        let shader = |name, stage, dxil| {
            crate::pipeline::test_shader(
                name,
                HostShader {
                    stage,
                    dxil,
                    group_base: None,
                    user_data: vec![],
                    samplers: vec![],
                    static_samplers: vec![],
                    root_cbvs: vec![],
                    root_srvs: vec![],
                    root_uavs: vec![],
                    stride_table: false,
                    run_time_buffers: 0,
                },
            )
        };
        let z = f32::from(class) * f32::from_bits(0x3780_0080);
        let vs = probe::vertex(
            &[[-1., -1., z, 1.], [-1., 3., z, 1.], [3., -1., z, 1.]],
            &[[1.; 4]; 3],
            &[1; 3],
        );
        let mut graphics = crate::pipeline::host::GraphicsState::default();
        graphics.topology = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        graphics.targets.NumRenderTargets = 1;
        graphics.targets.RTFormats[0] = DXGI_FORMAT(28);
        #[repr(C)]
        struct Depth(u32, D3D12_DEPTH_STENCIL_DESC);
        let test = Depth(
            11,
            D3D12_DEPTH_STENCIL_DESC {
                DepthEnable: true.into(),
                DepthWriteMask: D3D12_DEPTH_WRITE_MASK_ZERO,
                DepthFunc: D3D12_COMPARISON_FUNC_EQUAL,
                ..Default::default()
            },
        );
        #[repr(C)]
        struct Format(u32, u32);
        let format = Format(16, D16_UNORM_S8_UINT);
        // SAFETY: whole subobjects in the stream's layout.
        unsafe {
            graphics.read(11, (&raw const test).cast());
            graphics.read(16, (&raw const format).cast());
        }
        crate::pipeline::test_graphics_pipeline(
            vec![
                shader("vertex", HwStage::Vs, vs),
                shader("pixel", HwStage::Ps, probe::pixel()),
            ],
            graphics,
            &[0],
        )
    }

    /// The console's D16S8 depth, which the material resolve writes
    /// (classes in 16-bit steps) and the lighting tests.
    const D16_UNORM_S8_UINT: u32 = 118;

    /// The title's descriptor heaps, a view heap and a sampler heap, made as
    /// it makes them. Draws bind them through `draw::State`, as the title's
    /// do, which sets them again after a pass of ours. Bound on the host
    /// list behind `draw::State`'s back, as this module's test once did,
    /// they were lost to the pass: the draw after it set a root signature
    /// that indexes the sampler heap directly without one (the debug
    /// layer's message 1419).
    struct TitleHeaps {
        views: crate::descriptor::TestHeap,
        samplers: This,
    }

    impl TitleHeaps {
        fn new() -> TitleHeaps {
            /// The description and placement newer titles pass.
            #[repr(C)]
            struct Desc {
                kind: u32,
                count: u32,
                flags: u32,
                nodes: u32,
                address: u64,
                size: u64,
            }
            const HEAP_IID: crate::object::Iid = crate::object::guid(
                0x8efb471d,
                0x616c,
                0x4f49,
                [0x90, 0xf7, 0x12, 0x7b, 0xb7, 0x63, 0xfa, 0x51],
            );
            const SAMPLERS: u32 = 1;
            const SHADER_VISIBLE: u32 = 1;
            let desc = Desc {
                kind: SAMPLERS,
                count: 1,
                flags: SHADER_VISIBLE,
                nodes: 0,
                address: 0,
                size: 0,
            };
            let mut samplers = std::ptr::null();
            let made = crate::descriptor::create_heap(
                std::ptr::null(),
                (&raw const desc).cast(),
                &HEAP_IID,
                &mut samplers,
            );
            assert_eq!(made, crate::object::S_OK, "a title sampler heap");
            TitleHeaps {
                views: crate::descriptor::TestHeap::views(1),
                samplers,
            }
        }
    }

    impl Drop for TitleHeaps {
        fn drop(&mut self) {
            release(self.samplers);
        }
    }

    /// A 4×4 console D16S8 depth texture with its twins, RGBA8 targets, the
    /// title's descriptors of them (records in our memory, as its heaps
    /// hold them) and its heaps.
    struct Scene {
        depth: This,
        targets: Vec<This>,
        /// The handles of the targets' views, then of the depth's.
        handles: Vec<u64>,
        _records: Box<[[u64; 16]]>,
        heaps: TitleHeaps,
    }

    impl Scene {
        fn new(targets: usize) -> Scene {
            use crate::resource::tests_support::{committed, target};
            let null = std::ptr::null();
            let depth = depth(4, 4, D16_UNORM_S8_UINT);
            assert!(crate::resource::twin_view(depth, None).is_some());
            let targets: Vec<This> = (0..targets).map(|_| committed(&target(4, 4, 28))).collect();
            let mut records = vec![[0u64; 16]; targets.len() + 1].into_boxed_slice();
            let handles: Vec<u64> = records.iter_mut().map(|r| r.as_mut_ptr() as u64).collect();
            for (&target, &handle) in targets.iter().zip(&handles) {
                crate::descriptor::create_rtv(null, target, null.cast(), handle);
            }
            crate::descriptor::create_dsv(null, depth, null.cast(), handles[targets.len()]);
            Scene {
                depth,
                targets,
                handles,
                _records: records,
                heaps: TitleHeaps::new(),
            }
        }

        fn dsv(&self) -> u64 {
            self.handles[self.targets.len()]
        }

        /// Clears target `i` to black in `list`.
        fn clear(&self, list: &HostList, i: usize) {
            let host = crate::host::host().unwrap();
            let texture = crate::resource::host_resource(self.targets[i]).unwrap();
            crate::list::ensure_state(list, self.targets[i], 0x4);
            // SAFETY: an open list and a live RGBA8 target.
            unsafe {
                list.list
                    .ClearRenderTargetView(host.rtv(texture, None), &[0.; 4], None);
            }
        }

        /// The title's draw state in `list`, as it sets it: its heaps,
        /// target `i` (cleared to black) over the depth texture, the whole
        /// viewport, the whole scissor (none if `scissor` is false), a
        /// triangle list.
        fn state(&self, list: &HostList, i: usize, scissor: bool) -> crate::draw::State {
            self.clear(list, i);
            let mut state = crate::draw::State::default();
            state.set_heaps(&[self.heaps.views.0, self.heaps.samplers]);
            state.set_targets(list, &self.handles[i..=i], self.dsv());
            crate::draw::viewports(
                list,
                &[D3D12_VIEWPORT {
                    Width: 4.0,
                    Height: 4.0,
                    MaxDepth: 1.0,
                    ..Default::default()
                }],
            );
            if scissor {
                crate::draw::scissors(
                    list,
                    &[RECT {
                        right: 4,
                        bottom: 4,
                        ..Default::default()
                    }],
                );
            }
            state.set_topology(list, 4);
            state
        }

        /// The title's draw state, now on target `i` (cleared to black).
        fn target(&self, list: &HostList, state: &mut crate::draw::State, i: usize) {
            self.clear(list, i);
            state.set_targets(list, &self.handles[i..=i], self.dsv());
        }

        /// Which pixels of target `i` draws lit, row by row.
        fn lit(&self, i: usize) -> [[bool; 4]; 4] {
            let host = crate::host::host().unwrap();
            let texture = crate::resource::host_resource(self.targets[i]).unwrap();
            let image = crate::resource::tests_support::read_back(host, texture, 0x4);
            std::array::from_fn(|row| std::array::from_fn(|x| image[row * 256 + x * 4] == 255))
        }
    }

    impl Drop for Scene {
        fn drop(&mut self) {
            for &target in &self.targets {
                release(target);
            }
            release(self.depth);
        }
    }

    /// The pixels where `lit` holds (x, row), row by row.
    fn lit_where(lit: impl Fn(usize, usize) -> bool) -> [[bool; 4]; 4] {
        std::array::from_fn(|row| std::array::from_fn(|x| lit(x, row)))
    }

    /// The title draws with `pipeline`, as `state` has it set, in `list`.
    fn draw(list: &HostList, state: &mut crate::draw::State, pipeline: This) {
        state.set_pipeline(pipeline);
        state.draw(list, 3, 1, 0, 0);
    }

    /// The class each texel of the depth plane of 24-bit depth texture
    /// `depth` holds (its depth over the step of the lighting), row by row.
    fn classes(depth: This) -> [[u32; 4]; 4] {
        let copy = crate::resource::host_copy(depth).unwrap();
        let bytes = read(&copy.texture, copy.state.load(Ordering::Relaxed), 0);
        let step = f64::from(f32::from_bits(0x3780_0080));
        std::array::from_fn(|row| {
            std::array::from_fn(|x| {
                let at = row * 256 + x * 4;
                let word = u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap());
                let z = f64::from(word & 0xff_ffff) / f64::from(0xff_ffffu32);
                (z / step).round() as u32
            })
        })
    }

    /// The title's `ClearDepthStencilView` in `list` of the scene's depth
    /// plane to `z`, in `rects` (all of it if none).
    fn clear_depth(list: &crate::testgpu::TitleList, scene: &Scene, z: f32, rects: &[RECT]) {
        let this = list.this();
        type Clear = extern "system" fn(This, u64, u32, f32, u8, u32, *const RECT);
        const DEPTH: u32 = 1;
        crate::object::slot::<Clear>(this, 0x198)(
            this,
            scene.dsv(),
            DEPTH,
            z,
            0,
            rects.len() as u32,
            rects.as_ptr(),
        );
    }

    /// The tiled lighting: shaders write each pixel's class into the twin
    /// of the depth plane, then a draw of the same list tests a class
    /// "equal" against the plane. The plane holds the classes of the frame
    /// before (65 everywhere); this frame writes 129 on the left. The draw
    /// of class 129, bound by `draw::State` as the title's are, must light
    /// the left: the plane's pass comes before it (`planes_now`). Before,
    /// the pass waited for the list's close and the draw found only 65:
    /// nothing was lit (`XTP_PLANE_NOW=0` fails this test). The title's
    /// scissor (the top half), its target and its heaps are back after the
    /// pass.
    #[test]
    fn classes_written_in_a_list_are_tested_equal_in_the_same_list() {
        let test =
            "host::twins::tests::classes_written_in_a_list_are_tested_equal_in_the_same_list";
        let Some(_gpu) = crate::testgpu::device(test) else {
            return;
        };
        let scene = Scene::new(1);
        let plane = crate::resource::twin(scene.depth)
            .and_then(|t| t.plane.clone())
            .expect("a depth plane twin");
        // SAFETY: live textures.
        let (d, p) = unsafe {
            (
                crate::resource::host_resource(scene.depth)
                    .unwrap()
                    .GetDesc(),
                plane.texture.GetDesc(),
            )
        };
        // R24G8 typeless for the depth, R16 typeless for its plane's twin.
        assert_eq!((d.Format.0, p.Format.0), (44, 53));
        let pipeline = equal_pipeline(129);

        // The frame before: 65 everywhere, in the plane at the close.
        let before = crate::testgpu::TitleList::new();
        write_classes(before.host(), scene.depth, 65, 65);
        before.run();

        let frame = crate::testgpu::TitleList::new();
        let list = frame.host();
        write_classes(list, scene.depth, 129, 65);
        let mut state = scene.state(list, 0, false);
        crate::draw::scissors(
            list,
            &[RECT {
                right: 4,
                bottom: 2,
                ..Default::default()
            }],
        );
        draw(list, &mut state, pipeline);
        frame.run();

        assert_eq!(scene.lit(0), lit_where(|x, row| x < 2 && row < 2));
        release(pipeline);
    }

    /// Shaders write class 65 into the depth plane's twin, then the title
    /// clears the left half of the depth plane to 0 and draws "equal" 0,
    /// then 65. The pass comes before the clear (`twin_use`): 0 on the
    /// left, 65 on the right. Before, it waited for the next draw and drew
    /// 65 over the clear: no texel was 0.
    #[test]
    fn a_depth_clear_after_twin_writes_is_not_drawn_over() {
        let test = "host::twins::tests::a_depth_clear_after_twin_writes_is_not_drawn_over";
        let Some(_gpu) = crate::testgpu::device(test) else {
            return;
        };
        let scene = Scene::new(2);
        let (cleared, written) = (equal_pipeline(0), equal_pipeline(65));
        let before = crate::testgpu::TitleList::new();
        write_classes(before.host(), scene.depth, 33, 33);
        before.run();

        let frame = crate::testgpu::TitleList::new();
        let list = frame.host();
        write_classes(list, scene.depth, 65, 65);
        let left = RECT {
            right: 2,
            bottom: 4,
            ..Default::default()
        };
        clear_depth(&frame, &scene, 0.0, &[left]);
        let mut state = scene.state(list, 0, true);
        draw(list, &mut state, cleared);
        scene.target(list, &mut state, 1);
        draw(list, &mut state, written);
        frame.run();

        assert_eq!(scene.lit(0), lit_where(|x, _| x < 2), "the clear");
        assert_eq!(scene.lit(1), lit_where(|x, _| x >= 2), "the classes");
        release(cleared);
        release(written);
    }

    /// The title clears the depth plane to 0, then a shader writes the
    /// stencil plane through a view of plane 1, then a draw tests "equal"
    /// 0. Only the stencil plane's twin is ahead (`twin_view_written`), so
    /// no pass comes before the draw and every pixel is lit. Before, both
    /// twins counted as written, and the pass drew the depth plane's twin
    /// (the frame before's 33) over the clear: nothing was lit.
    #[test]
    fn a_stencil_write_leaves_the_depth_plane_as_the_list_had_it() {
        let test = "host::twins::tests::a_stencil_write_leaves_the_depth_plane_as_the_list_had_it";
        let Some(_gpu) = crate::testgpu::device(test) else {
            return;
        };
        let scene = Scene::new(1);
        let cleared = equal_pipeline(0);
        let before = crate::testgpu::TitleList::new();
        write_classes(before.host(), scene.depth, 33, 33);
        before.run();

        let frame = crate::testgpu::TitleList::new();
        let list = frame.host();
        clear_depth(&frame, &scene, 0.0, &[]);
        // A 2D view of plane 1 (the stencil), as the title describes it.
        let stencil = crate::descriptor::View::Unordered(
            scene.depth as usize,
            0,
            Some([0, 4, 0, 1, 0, 0, 0, 0, 0, 0]),
        );
        assert!(list.twin_use(scene.depth, true, true));
        list.twin_view_written(scene.depth, &stencil);
        let twin = crate::resource::twin(scene.depth).unwrap();
        list.move_to(&twin.colour, COPY_DEST);
        fill(list, &twin.colour.texture, 0, &[7; 1024]);
        list.move_to(&twin.colour, UNORDERED_ACCESS);
        list.twins_written();
        let mut state = scene.state(list, 0, true);
        draw(list, &mut state, cleared);
        frame.run();

        assert_eq!(scene.lit(0), [[true; 4]; 4]);
        release(cleared);
    }

    /// Shaders write classes 129 (left) and 65 into the depth plane's twin,
    /// then the title copies the depth texture into another: the copy has
    /// them, the pass coming before it (`twin_use`). Before, the copy came
    /// first and took the frame before's 65 everywhere.
    #[test]
    fn a_copy_of_the_depth_texture_takes_the_classes_written_before_it() {
        let test =
            "host::twins::tests::a_copy_of_the_depth_texture_takes_the_classes_written_before_it";
        let Some(_gpu) = crate::testgpu::device(test) else {
            return;
        };
        let scene = Scene::new(0);
        let other = depth(4, 4, D16_UNORM_S8_UINT);
        let before = crate::testgpu::TitleList::new();
        write_classes(before.host(), scene.depth, 65, 65);
        before.run();

        let frame = crate::testgpu::TitleList::new();
        write_classes(frame.host(), scene.depth, 129, 65);
        let this = frame.this();
        crate::object::slot::<extern "system" fn(This, This, This)>(this, 0x88)(
            this,
            other,
            scene.depth,
        );
        frame.run();

        assert_eq!(classes(other), [[129, 129, 65, 65]; 4]);
        release(other);
    }

    /// A list draws one depth plane from its twin (65), and before it runs
    /// another list draws a second plane 300 times (129). Each plane's
    /// passes read views of their own, which last as long as the lists
    /// that read them: the first list's plane gets 65. Before, the views
    /// were a ring of 256 slots that every pass took the next of: the
    /// first list's slot held a view of the second plane when it ran.
    #[test]
    fn a_passs_view_lasts_until_its_list_has_run() {
        let test = "host::twins::tests::a_passs_view_lasts_until_its_list_has_run";
        let Some(gpu) = crate::testgpu::device(test) else {
            return;
        };
        let host = gpu.host();
        let (a, b) = (
            depth(4, 4, D16_UNORM_S8_UINT),
            depth(4, 4, D16_UNORM_S8_UINT),
        );
        for d in [a, b] {
            assert!(crate::resource::twin_view(d, None).is_some());
        }
        let first = host.new_list().unwrap();
        write_classes(&first, a, 65, 65);
        assert!(first.planes_now(), "a pass");
        let second = host.new_list().unwrap();
        write_classes(&second, b, 129, 129);
        let (copy, twin) = (
            crate::resource::host_copy(b).unwrap(),
            crate::resource::twin(b).unwrap(),
        );
        for _ in 0..300 {
            second.pass_plane(&copy, &twin);
        }
        second.close();
        first.close();
        host.execute(&[&first]);
        host.drain();
        assert_eq!(classes(a), [[65; 4]; 4], "the first plane");
        host.execute(&[&second]);
        host.drain();
        assert_eq!(classes(b), [[129; 4]; 4], "the second plane");
        release(a);
        release(b);
    }

    /// A dispatch writes the depth plane's twin through a view of the
    /// title's, which also asks unordered access of the depth texture (the
    /// common state on the host: `legal_state`); a pass comes before the
    /// next dispatch. The list then closes: the depth texture is left
    /// readable by shaders, for lists that pick it from a heap by a
    /// computed index. Before, the pass put it back in the common state and
    /// the close, finding nothing to draw, left it there.
    #[test]
    fn a_plane_drawn_before_the_close_is_left_readable() {
        let test = "host::twins::tests::a_plane_drawn_before_the_close_is_left_readable";
        let Some(gpu) = crate::testgpu::device(test) else {
            return;
        };
        let host = gpu.host();
        let depth = depth(4, 4, D16_UNORM_S8_UINT);
        assert!(crate::resource::twin_view(depth, None).is_some());
        let list = host.new_list().unwrap();
        assert!(list.twin_use(depth, true, true));
        crate::list::ensure_state(&list, depth, UNORDERED_ACCESS);
        let plane = crate::resource::twin(depth).unwrap().plane.clone().unwrap();
        let mut steps = vec![0u8; 1024];
        for row in 0..4 {
            for x in 0..4 {
                steps[row * 256 + x * 2..][..2].copy_from_slice(&65u16.to_le_bytes());
            }
        }
        list.move_to(&plane, COPY_DEST);
        fill(&list, &plane.texture, 0, &steps);
        list.move_to(&plane, UNORDERED_ACCESS);
        list.twins_written();
        // The next dispatch.
        assert!(list.planes_now(), "a pass");
        list.close();
        host.execute(&[&list]);
        host.drain();
        let copy = crate::resource::host_copy(depth).unwrap();
        let readable = super::super::legal_state(&copy.texture, SHADER_RESOURCE);
        assert_eq!(copy.state.load(Ordering::Relaxed), readable);
        assert_eq!(classes(depth), [[65; 4]; 4]);
        release(depth);
    }

    const OCCLUSION: u32 = 0;

    /// A title heap of `count` occlusion queries.
    fn occlusion_heap(count: u32) -> This {
        #[repr(C)]
        struct Desc {
            kind: u32,
            count: u32,
            nodes: u32,
        }
        const QUERY_HEAP_IID: crate::object::Iid = crate::object::guid(
            0x0d9658ae,
            0xed45,
            0x469e,
            [0xa6, 0x1d, 0x97, 0x0e, 0xc5, 0x83, 0xca, 0xb4],
        );
        let mut heap = std::ptr::null();
        let desc = Desc {
            kind: OCCLUSION,
            count,
            nodes: 0,
        };
        assert_eq!(
            crate::query::create(
                std::ptr::null(),
                (&raw const desc).cast(),
                &QUERY_HEAP_IID,
                &mut heap
            ),
            crate::object::S_OK
        );
        heap
    }

    /// The title's BeginQuery (`open`) or EndQuery of occlusion query
    /// `index` of `heap` in `list`.
    fn query(list: &crate::testgpu::TitleList, heap: This, index: u32, open: bool) {
        let this = list.this();
        let offset = if open { 0xe0 } else { 0xe8 };
        crate::object::slot::<extern "system" fn(This, This, u32, u32)>(this, offset)(
            this, heap, OCCLUSION, index,
        );
    }

    /// The title's ResolveQueryData of the first `count` occlusion
    /// queries of `heap` into `out`, in `list`; `out` starts as 0xface
    /// words.
    fn resolve(
        list: &crate::testgpu::TitleList,
        heap: This,
        count: u32,
        out: &crate::testgpu::TitleBuffer,
    ) {
        out.set_words(|_| 0xface);
        let this = list.this();
        crate::object::slot::<extern "system" fn(This, This, u32, u32, u32, This, u64)>(this, 0xf0)(
            this,
            heap,
            OCCLUSION,
            0,
            count,
            out.resource(),
            0,
        );
    }

    /// Two occlusion queries of the title's. The first is around a draw
    /// that passes no sample ("equal" 7 against classes 65): the classes
    /// written before it are drawn into the plane as it opens, and it
    /// counts 0. In the second, shaders write classes 129 (left) and 65,
    /// then a draw tests "equal" 129: it lights the left, and the query
    /// counts those 8 samples only. The plane is copied in the query
    /// (`copy_plane`), which counts nothing. In round 2, the pass waited
    /// for the query's end: the draw found 65 everywhere, nothing was lit
    /// and the query counted 0 (the Codex review's reproduction). Before,
    /// the passes came before the draws, in the queries, and each counted
    /// the plane's 16 texels.
    #[test]
    fn a_draw_in_a_query_tests_the_classes_written_in_it() {
        let test = "host::twins::tests::a_draw_in_a_query_tests_the_classes_written_in_it";
        let Some(_gpu) = crate::testgpu::device(test) else {
            return;
        };
        let heap = occlusion_heap(2);
        let out = crate::testgpu::TitleBuffer::new(256);
        let scene = Scene::new(1);
        let (none, lit) = (equal_pipeline(7), equal_pipeline(129));
        let before = crate::testgpu::TitleList::new();
        write_classes(before.host(), scene.depth, 33, 33);
        before.run();

        let frame = crate::testgpu::TitleList::new();
        let list = frame.host();
        write_classes(list, scene.depth, 65, 65);
        let mut state = scene.state(list, 0, true);
        query(&frame, heap, 0, true);
        draw(list, &mut state, none);
        query(&frame, heap, 0, false);
        query(&frame, heap, 1, true);
        write_classes(list, scene.depth, 129, 65);
        draw(list, &mut state, lit);
        query(&frame, heap, 1, false);
        resolve(&frame, heap, 2, &out);
        frame.run();

        assert_eq!(
            out.words()[..4],
            [0, 0, 8, 0],
            "samples counted by the queries"
        );
        assert_eq!(scene.lit(0), lit_where(|x, _| x < 2));
        release(none);
        release(lit);
        release(heap);
    }

    /// In an occlusion query of the title's, shaders write classes 129
    /// (left) and 65 into the depth plane's twin, then the title clears the
    /// left half of the depth plane to 0; after the query, draws test
    /// "equal" 0, then 65. The plane is copied before the clear, in the
    /// query (`copy_plane`): 0 on the left, 65 on the right, and the query
    /// counts nothing. In round 2, the pass waited for the query's end and
    /// drew 129 and 65 over the clear: no texel was 0 (the Codex review's
    /// case 1 in a query).
    #[test]
    fn a_depth_clear_in_a_query_is_not_drawn_over() {
        let test = "host::twins::tests::a_depth_clear_in_a_query_is_not_drawn_over";
        let Some(_gpu) = crate::testgpu::device(test) else {
            return;
        };
        let heap = occlusion_heap(1);
        let out = crate::testgpu::TitleBuffer::new(256);
        let scene = Scene::new(2);
        let (cleared, written) = (equal_pipeline(0), equal_pipeline(65));
        let before = crate::testgpu::TitleList::new();
        write_classes(before.host(), scene.depth, 33, 33);
        before.run();

        let frame = crate::testgpu::TitleList::new();
        let list = frame.host();
        query(&frame, heap, 0, true);
        write_classes(list, scene.depth, 129, 65);
        let left = RECT {
            right: 2,
            bottom: 4,
            ..Default::default()
        };
        clear_depth(&frame, &scene, 0.0, &[left]);
        query(&frame, heap, 0, false);
        resolve(&frame, heap, 1, &out);
        let mut state = scene.state(list, 0, true);
        draw(list, &mut state, cleared);
        scene.target(list, &mut state, 1);
        draw(list, &mut state, written);
        frame.run();

        assert_eq!(out.words()[..2], [0; 2], "samples counted by the query");
        assert_eq!(scene.lit(0), lit_where(|x, _| x < 2), "the clear");
        assert_eq!(scene.lit(1), lit_where(|x, _| x >= 2), "the classes");
        assert_eq!(classes(scene.depth), [[0, 0, 65, 65]; 4]);
        release(cleared);
        release(written);
        release(heap);
    }

    /// In an occlusion query of the title's, shaders write classes 129
    /// (left) and 65 into the depth plane's twin, then the title copies the
    /// depth texture into another: the copy has them, the plane copied
    /// before it in the query (`copy_plane`), which counts nothing. In
    /// round 2, the pass waited for the query's end: the copy took the
    /// frame before's 65 everywhere (the Codex review's case 3 in a query).
    #[test]
    fn a_copy_in_a_query_takes_the_classes_written_before_it() {
        let test = "host::twins::tests::a_copy_in_a_query_takes_the_classes_written_before_it";
        let Some(_gpu) = crate::testgpu::device(test) else {
            return;
        };
        let heap = occlusion_heap(1);
        let out = crate::testgpu::TitleBuffer::new(256);
        let scene = Scene::new(0);
        let other = depth(4, 4, D16_UNORM_S8_UINT);
        let before = crate::testgpu::TitleList::new();
        write_classes(before.host(), scene.depth, 65, 65);
        before.run();

        let frame = crate::testgpu::TitleList::new();
        query(&frame, heap, 0, true);
        write_classes(frame.host(), scene.depth, 129, 65);
        let this = frame.this();
        crate::object::slot::<extern "system" fn(This, This, This)>(this, 0x88)(
            this,
            other,
            scene.depth,
        );
        query(&frame, heap, 0, false);
        resolve(&frame, heap, 1, &out);
        frame.run();

        assert_eq!(out.words()[..2], [0; 2], "samples counted by the query");
        assert_eq!(classes(other), [[129, 129, 65, 65]; 4]);
        release(other);
        release(heap);
    }

    /// Records into `list` the copy of `texels` (rows of `row` bytes, one
    /// after the other) into subresource `index` of `texture`, in the copy
    /// destination state, laid out as the host copies it.
    fn fill_rows(list: &HostList, texture: &ID3D12Resource, index: u32, texels: &[u8], row: usize) {
        let host = host().unwrap();
        // SAFETY: a live texture.
        let d = unsafe { texture.GetDesc() };
        let mut footprint = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
        let mut bytes = 0;
        // SAFETY: a plain query for one subresource.
        unsafe {
            host.device.GetCopyableFootprints(
                &d,
                index,
                1,
                0,
                Some(&mut footprint),
                None,
                None,
                Some(&mut bytes),
            );
        }
        let (upload, data) = host.upload_buffer(bytes).unwrap();
        let pitch = footprint.Footprint.RowPitch as usize;
        for (y, texels) in texels.chunks(row).enumerate() {
            // SAFETY: the buffer's mapping holds the footprint's rows.
            unsafe {
                std::ptr::copy_nonoverlapping(texels.as_ptr(), data.add(y * pitch), texels.len());
            }
        }
        footprint.Offset = 0;
        // SAFETY: an open list and live resources.
        unsafe {
            list.list.CopyTextureRegion(
                &subresource_at(texture, index),
                0,
                0,
                0,
                &placed_at(&upload, &footprint),
                None,
            );
        }
        list.keep(&upload);
    }

    /// Subresource `index` of `texture` (in `state`): its rows, one after
    /// the other, each `row` bytes long.
    fn read_rows(texture: &ID3D12Resource, state: u32, index: u32, row: usize) -> Vec<u8> {
        let host = host().unwrap();
        // SAFETY: a live texture.
        let d = unsafe { texture.GetDesc() };
        let mut footprint = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
        let mut bytes = 0;
        // SAFETY: a plain query for one subresource.
        unsafe {
            host.device.GetCopyableFootprints(
                &d,
                index,
                1,
                0,
                Some(&mut footprint),
                None,
                None,
                Some(&mut bytes),
            );
        }
        footprint.Offset = 0;
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_READBACK,
            ..Default::default()
        };
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
        let mut readback: Option<ID3D12Resource> = None;
        // SAFETY: plain creation of a read-back buffer.
        unsafe {
            host.device
                .CreateCommittedResource(
                    &heap,
                    D3D12_HEAP_FLAG_NONE,
                    &desc,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    None,
                    &mut readback,
                )
                .unwrap();
        }
        let readback = readback.unwrap();
        let list = host.new_list().unwrap();
        let state = D3D12_RESOURCE_STATES(super::super::legal_state(texture, state) as i32);
        let source = D3D12_RESOURCE_STATE_COPY_SOURCE;
        // SAFETY: an open list and live resources.
        unsafe {
            if state != source {
                list.list
                    .ResourceBarrier(&[transition(texture, state, source)]);
            }
            list.list.CopyTextureRegion(
                &placed_at(&readback, &footprint),
                0,
                0,
                0,
                &subresource_at(texture, index),
                None,
            );
            if state != source {
                list.list
                    .ResourceBarrier(&[transition(texture, source, state)]);
            }
        }
        list.close();
        host.execute(&[&list]);
        host.drain();
        let pitch = footprint.Footprint.RowPitch as usize;
        let mut rows = Vec::with_capacity(row * footprint.Footprint.Height as usize);
        let mut mapped = std::ptr::null_mut();
        // SAFETY: the copy ran (drained); the buffer holds the footprint.
        unsafe {
            readback.Map(0, None, Some(&mut mapped)).unwrap();
            let all = std::slice::from_raw_parts(mapped.cast::<u8>(), bytes as usize);
            for y in 0..footprint.Footprint.Height as usize {
                rows.extend_from_slice(&all[y * pitch..][..row]);
            }
            readback.Unmap(0, None);
        }
        rows
    }

    /// A 256×256 title depth texture of console format `format` with a
    /// depth plane twin holding `texels` (4 or 2 bytes each, row by row):
    /// its plane drawn from the twin by the pass, and the plane of another
    /// such texture copied from the same twin as in a query
    /// (`copy_plane`), both read back, 4 bytes a texel.
    fn drawn_and_copied(format: u32, texels: &[u8]) -> (Vec<u8>, Vec<u8>) {
        let host = host().unwrap();
        let (drawn, copied) = (depth(256, 256, format), depth(256, 256, format));
        let list = host.new_list().unwrap();
        for (d, copy) in [(drawn, false), (copied, true)] {
            assert!(crate::resource::twin_view(d, None).is_some());
            let (depth, twin) = (
                crate::resource::host_copy(d).unwrap(),
                crate::resource::twin(d).unwrap(),
            );
            let plane = twin.plane.clone().expect("a depth plane twin");
            list.move_to(&plane, COPY_DEST);
            fill_rows(&list, &plane.texture, 0, texels, texels.len() / 256);
            list.move_to(&plane, UNORDERED_ACCESS);
            if copy {
                assert!(list.copy_plane(&depth, &twin), "a copy");
            } else {
                list.pass_plane(&depth, &twin);
            }
        }
        list.close();
        host.execute(&[&list]);
        host.drain();
        let [drawn_plane, copied_plane] = [drawn, copied].map(|d| {
            let copy = crate::resource::host_copy(d).unwrap();
            read_rows(
                &copy.texture,
                copy.state.load(Ordering::Relaxed),
                0,
                256 * 4,
            )
        });
        release(drawn);
        release(copied);
        (drawn_plane, copied_plane)
    }

    /// The texels where two planes of 24-bit depth (4 bytes a texel, the
    /// depth in the low 24 bits) differ: (texel, first, second).
    fn differ_24(a: &[u8], b: &[u8]) -> Vec<(usize, u32, u32)> {
        let word = |bytes: &[u8], at: usize| {
            u32::from_le_bytes(bytes[at * 4..at * 4 + 4].try_into().unwrap()) & 0xff_ffff
        };
        (0..a.len() / 4)
            .map(|i| (i, word(a, i), word(b, i)))
            .filter(|(_, x, y)| x != y)
            .collect()
    }

    /// Every 16-bit step (0 to 65535, texel `y * 256 + x`) copied into a
    /// 24-bit plane, as in a query, gets the very depth the pass draws for
    /// it: the table holds what the host's depth writes give each step's
    /// float (`PlanePass::steps_table`), which the lighting's "equal" draws
    /// get too.
    #[test]
    fn a_plane_copied_in_a_query_holds_what_the_pass_draws_for_each_step() {
        let test =
            "host::twins::tests::a_plane_copied_in_a_query_holds_what_the_pass_draws_for_each_step";
        let Some(_gpu) = crate::testgpu::device(test) else {
            return;
        };
        let steps: Vec<u8> = (0..=u16::MAX).flat_map(u16::to_le_bytes).collect();
        let (drawn, copied) = drawn_and_copied(D16_UNORM_S8_UINT, &steps);
        let differ = differ_24(&drawn, &copied);
        assert!(
            differ.is_empty(),
            "{} steps differ, first (step, drawn, copied): {:?}",
            differ.len(),
            &differ[..differ.len().min(8)]
        );
    }

    /// Floats spread over the depths a plane holds (below 0, 0 to 1 at
    /// every exponent from 2^-31, the lighting's class steps, 1 and above),
    /// 65536 of them, 4 bytes each.
    fn spread_floats() -> Vec<u8> {
        let step = f32::from_bits(0x3780_0080);
        let mut seed = 0x2545_f491_u32;
        let floats: Vec<f32> = (0..65536u32)
            .map(|i| match i {
                0 => -0.25,
                1 => 0.0,
                2 => 1.0,
                3 => 1.5,
                4 => f32::from_bits(0x3f7f_ffff),
                5..=4100 => (i - 5) as f32 * step,
                _ => {
                    seed ^= seed << 13;
                    seed ^= seed >> 17;
                    seed ^= seed << 5;
                    f32::from_bits(0x3000_0000 + seed % (0x3f80_0000 - 0x3000_0000))
                }
            })
            .collect();
        floats.iter().flat_map(|f| f.to_le_bytes()).collect()
    }

    /// Floats copied into a 24-bit plane, as in a query, get the depth the
    /// pass draws for them on WARP ([`UNORM24`]). With D3D's rule computed
    /// exactly, 992 of these floats got one step more.
    #[test]
    fn a_24_bit_plane_copied_in_a_query_holds_what_the_pass_draws_for_floats() {
        let test = "host::twins::tests::a_24_bit_plane_copied_in_a_query_holds_what_the_pass_draws_for_floats";
        let Some(_gpu) = crate::testgpu::device(test) else {
            return;
        };
        const D24_UNORM_S8_UINT: u32 = 45;
        let (drawn, copied) = drawn_and_copied(D24_UNORM_S8_UINT, &spread_floats());
        let differ = differ_24(&drawn, &copied);
        assert!(
            differ.is_empty(),
            "{} floats differ, first (texel, drawn, copied): {:?}",
            differ.len(),
            &differ[..differ.len().min(8)]
        );
    }

    /// Floats copied into a 32-bit float plane, as in a query, are what the
    /// pass draws for them: the same floats, from 0 to 1.
    #[test]
    fn a_float_plane_copied_in_a_query_holds_what_the_pass_draws() {
        let test = "host::twins::tests::a_float_plane_copied_in_a_query_holds_what_the_pass_draws";
        let Some(_gpu) = crate::testgpu::device(test) else {
            return;
        };
        const D32_FLOAT_S8X24_UINT: u32 = 20;
        let (drawn, copied) = drawn_and_copied(D32_FLOAT_S8X24_UINT, &spread_floats());
        let differ: Vec<usize> = (0..drawn.len() / 4)
            .filter(|i| drawn[i * 4..i * 4 + 4] != copied[i * 4..i * 4 + 4])
            .collect();
        assert!(
            differ.is_empty(),
            "{} floats differ, first texels: {:?}",
            differ.len(),
            &differ[..differ.len().min(8)]
        );
    }

    /// The title sets a viewport and no scissor, then draws: with a pass
    /// before the draw, the draw finds the same as without one, the
    /// list's own lack of scissor. Before, the pass left its own scissor
    /// over the whole target in place.
    #[test]
    fn a_pass_leaves_no_scissor_where_the_title_set_none() {
        let test = "host::twins::tests::a_pass_leaves_no_scissor_where_the_title_set_none";
        let Some(_gpu) = crate::testgpu::device(test) else {
            return;
        };
        let scene = Scene::new(2);
        let pipeline = equal_pipeline(65);
        let before = crate::testgpu::TitleList::new();
        write_classes(before.host(), scene.depth, 65, 65);
        before.run();

        let plain = crate::testgpu::TitleList::new();
        let mut state = scene.state(plain.host(), 0, false);
        draw(plain.host(), &mut state, pipeline);
        plain.run();
        let passed = crate::testgpu::TitleList::new();
        write_classes(passed.host(), scene.depth, 65, 65);
        let mut state = scene.state(passed.host(), 1, false);
        draw(passed.host(), &mut state, pipeline);
        passed.run();

        assert_eq!(scene.lit(1), scene.lit(0));
        release(pipeline);
    }

    /// `XTP_PLANE_PROBE`: a probe of a pass is read once the GPU has run
    /// its list's submission, not before (whatever the time since), and
    /// finds the classes drawn.
    #[cfg(feature = "diagnostics")]
    #[test]
    fn a_probe_is_read_once_its_list_has_run() {
        let test = "host::twins::tests::a_probe_is_read_once_its_list_has_run";
        let Some(gpu) = crate::testgpu::device(test) else {
            return;
        };
        let host = gpu.host();
        let depth = depth(4, 4, D16_UNORM_S8_UINT);
        assert!(crate::resource::twin_view(depth, None).is_some());
        let list = host.new_list().unwrap();
        write_classes(&list, depth, 65, 129);
        assert!(list.planes_now(), "a pass");
        let (copy, twin) = (
            crate::resource::host_copy(depth).unwrap(),
            crate::resource::twin(depth).unwrap(),
        );
        probe::record_in(&list, &copy, &twin, true, 1);
        assert!(
            probe::ready_of(u64::MAX, Some(1)).is_empty(),
            "read before it ran"
        );
        list.close();
        host.execute(&[&list]);
        host.drain();
        let ready = probe::ready_of(host.completed(), Some(1));
        assert_eq!(ready.len(), 1);
        let found = probe::found(&ready[0]).unwrap();
        assert_eq!((found.differ, found.texels), (0, 16));
        assert_eq!(found.classes, [(65, 8), (129, 8)]);
        release(depth);
    }

    /// `XTP_PLANE_PROBE`: a closed list submitted a second time before its
    /// probe is read writes the probe's buffers again, so the probe waits
    /// for that run: it is not read at the first run's fence, and is read
    /// once after the second. In round 2, the first submission took the
    /// probe from the list with its fence, and the second was not seen (the
    /// Codex review's case 9, closed list submitted again).
    #[cfg(feature = "diagnostics")]
    #[test]
    fn a_probe_waits_for_its_lists_last_run() {
        let test = "host::twins::tests::a_probe_waits_for_its_lists_last_run";
        let Some(gpu) = crate::testgpu::device(test) else {
            return;
        };
        const FRAME: u64 = 2;
        let host = gpu.host();
        let depth = depth(4, 4, D16_UNORM_S8_UINT);
        assert!(crate::resource::twin_view(depth, None).is_some());
        let list = host.new_list().unwrap();
        write_classes(&list, depth, 65, 129);
        assert!(list.planes_now(), "a pass");
        let (copy, twin) = (
            crate::resource::host_copy(depth).unwrap(),
            crate::resource::twin(depth).unwrap(),
        );
        probe::record_in(&list, &copy, &twin, true, FRAME);
        list.close();
        host.execute(&[&list]);
        host.drain();
        let first = host.completed();
        host.execute(&[&list]);
        assert!(
            probe::ready_of(first, Some(FRAME)).is_empty(),
            "read at the first run's fence"
        );
        host.drain();
        let ready = probe::ready_of(host.completed(), Some(FRAME));
        assert_eq!(ready.len(), 1, "read once");
        let found = probe::found(&ready[0]).unwrap();
        assert_eq!((found.differ, found.texels), (0, 16));
        release(depth);
    }

    /// `XTP_PLANE_PROBE`: a 24-bit plane drawn from a twin of floats holds
    /// each float's nearest 24-bit value, which the probe counts as the
    /// same. Before, it compared the bits of the two floats: most texels
    /// differed.
    #[cfg(feature = "diagnostics")]
    #[test]
    fn a_24_bit_plane_matches_the_floats_it_was_drawn_from() {
        let step = f32::from_bits(0x3780_0080);
        let max = f64::from(0xff_ffffu32);
        for class in [1u16, 65, 129, 4000, 65535] {
            let float = f32::from(class) * step;
            let z = (f64::from(float) * max).round() / max;
            assert!(probe::same_float(z, true, float.to_bits()), "class {class}");
        }
        assert!(!probe::same_float(0.5, true, 0.25f32.to_bits()));
        assert!(probe::same_float(0.25, false, 0.25f32.to_bits()));
    }

    /// A depth texture with a stencil plane: what shaders write into the
    /// twins of its two planes is in the planes once the list closes, the
    /// depth plane drawn by the pass (class steps 65 and 129 times the
    /// float nearest 1/65535, as the lighting draws them), the stencil
    /// plane copied.
    #[test]
    fn a_depth_stencil_textures_twins_reach_its_planes() {
        let test = "host::twins::tests::a_depth_stencil_textures_twins_reach_its_planes";
        let Some(gpu) = crate::testgpu::device(test) else {
            return;
        };
        let host = gpu.host();
        const D32_FLOAT_S8X24_UINT: u32 = 20;
        let depth = depth(4, 4, D32_FLOAT_S8X24_UINT);
        assert!(crate::resource::twin_view(depth, None).is_some());
        let twin = crate::resource::twin(depth).expect("a twin");
        let plane = twin.plane.clone().expect("a depth plane twin");
        let step = f32::from_bits(0x3780_0080);
        let classes = [65.0 * step, 129.0 * step];
        let mut floats = vec![0u8; 1024];
        let mut stencil = vec![0u8; 1024];
        for row in 0..4 {
            for x in 0..4 {
                let v = classes[(row + x) % 2];
                floats[row * 256 + x * 4..][..4].copy_from_slice(&v.to_le_bytes());
                stencil[row * 256 + x] = (10 + row * 4 + x) as u8;
            }
        }
        let list = host.new_list().unwrap();
        assert!(list.twin_use(depth, true, true));
        for (texture, bytes) in [(&plane, &floats), (&twin.colour, &stencil)] {
            list.move_to(texture, COPY_DEST);
            fill(&list, &texture.texture, 0, bytes);
            list.move_to(texture, UNORDERED_ACCESS);
        }
        list.twins_written();
        list.close();
        host.execute(&[&list]);
        host.drain();
        let copy = crate::resource::host_copy(depth).unwrap();
        let state = copy.state.load(Ordering::Relaxed);
        let depths = read(&copy.texture, state, 0);
        let stencils = read(&copy.texture, state, 1);
        for row in 0..4 {
            for x in 0..4 {
                let at = row * 256 + x * 4;
                let got = f32::from_le_bytes(depths[at..at + 4].try_into().unwrap());
                assert_eq!(got, classes[(row + x) % 2], "depth at {x},{row}");
                let want = (10 + row * 4 + x) as u8;
                assert_eq!(stencils[row * 256 + x], want, "stencil at {x},{row}");
            }
        }
        release(depth);
    }
}
