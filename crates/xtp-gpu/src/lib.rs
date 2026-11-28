//! D3D12.X on host D3D12.
//!
//! D3D12.X records commands from code inlined into the title, which writes
//! packets into the command list while its put pointer is below a limit and
//! calls the list's method otherwise. Our lists keep the limits at zero, so
//! every command arrives here as a call, to be run on host D3D12. The
//! title's GPU addresses are CPU addresses (console memory is unified).
//!
//! [`device`] is the device object titles create, [`dxgi`] its adapter side,
//! [`resource`] heaps and resources, [`fence`] fences, [`queue`] command
//! queues, [`list`] command allocators and lists, [`descriptor`]
//! descriptor heaps, [`root`] root signatures, [`query`] query heaps,
//! [`signature`] command signatures, [`pipeline`] pipeline states,
//! [`frame`] frame pacing, [`object`] what all console graphics objects
//! share, [`console`] which console model the title sees. Planned:
//! `present` (swap chain). In tests, `testgpu` gives GPU tests the host
//! device (hardware, or WARP with `XTP_TEST_ADAPTER=warp`).

#[cfg(windows)]
mod agility;
#[cfg(windows)]
pub mod ambient;
#[cfg(all(windows, feature = "diagnostics"))]
mod capture;
pub mod console;
#[cfg(all(windows, feature = "diagnostics"))]
mod copystat;
#[cfg(windows)]
mod cores;
pub mod descriptor;
pub mod device;
#[cfg(windows)]
mod draw;
#[cfg(windows)]
mod dred;
pub mod dxgi;
pub mod fence;
pub mod frame;
#[cfg(all(windows, feature = "diagnostics"))]
mod frameprobe;
#[cfg(windows)]
mod host;
#[cfg(windows)]
mod indirect;
pub mod list;
#[cfg(windows)]
mod loopstat;
#[cfg(all(windows, feature = "diagnostics"))]
pub mod meshstat;
pub mod object;
#[cfg(all(windows, feature = "diagnostics"))]
mod passdump;
pub mod pipeline;
#[cfg(windows)]
mod pipestats;
#[cfg(windows)]
mod primitive;
pub mod query;
#[cfg(windows)]
mod readmap;
#[cfg(all(windows, any(test, feature = "diagnostics")))]
pub mod record;
#[cfg(windows)]
pub use readmap::{configure_copy_back, save_copy_back_map};
#[cfg(windows)]
pub use watch::{
    CopyBackPin, pin_copy_back, pin_title_read, set_copy_back_lookup, title_virtual_protect,
};
#[cfg(all(windows, feature = "readmap-test"))]
#[doc(hidden)]
pub fn publish_copy_back_for_test(at: usize, bytes: &mut [u8], writer: u64) -> Option<usize> {
    assert!(bytes.len() <= watch::PAGE - at % watch::PAGE);
    let key = readmap::Key {
        size: watch::PAGE as u64,
        alignment: 16,
        writer,
        offset: 0,
    };
    watch::publish(at, key, bytes, None).map(|(copied, _)| copied)
}
pub mod queue;
#[cfg(windows)]
mod rectangles;
pub mod resource;
pub mod root;
#[cfg(windows)]
pub mod rt;
#[cfg(windows)]
mod rt_standin;
#[cfg(windows)]
mod sampler;
#[cfg(windows)]
pub mod shader_cache;
#[cfg(windows)]
mod shadow;
pub mod signature;
#[cfg(windows)]
mod tdr;
#[cfg(all(windows, test))]
mod testgpu;
#[cfg(windows)]
mod texback;
#[cfg(windows)]
pub use texback::configure_texture_readback;
#[cfg(windows)]
mod texture;
#[cfg(all(windows, test))]
mod texture_commit_probe;
#[cfg(windows)]
mod timestamps;
#[cfg(windows)]
pub mod uma;
#[cfg(windows)]
mod watch;

#[cfg(windows)]
pub use host::on_failure;
#[cfg(windows)]
pub use watch::{
    KernelWrite, copy_asks, guarding, kernel_writing as kernel_will_write, set_alias_lookup,
    title_memory_remapped, writing as kernel_wrote,
};
