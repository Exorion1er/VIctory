//! Lot 7, geometry and textures that come and go: the switches of its
//! behaviour changes (each on by default, `VICTORY_L7_<NAME>=0` turns it
//! off) and the per-frame counts of what they did.

use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

/// Whether switch `name` (`VICTORY_L7_<name>`) is on: unset or anything but
/// `0`.
fn on(name: &str) -> bool {
    std::env::var_os(format!("VICTORY_L7_{name}")).is_none_or(|v| v != "0")
}

const NAMES: [&str; 5] = ["DYNAMIC_SLOTS", "CB_SLOTS", "VIEW_COPY_LRU", "VIEW_ADDRESS", "TEXTURE_SLOTS"];

/// The switches, read once: (name, on).
fn switches() -> &'static [(&'static str, bool); NAMES.len()] {
    static S: std::sync::OnceLock<[(&str, bool); NAMES.len()]> = std::sync::OnceLock::new();
    S.get_or_init(|| NAMES.map(|n| (n, on(n))))
}

/// `VICTORY_L7_DYNAMIC_SLOTS`: on a host that takes view heaps of
/// `UNCAPPED_VIEWS`, the dynamic slots (tables made afresh per draw) are
/// every slot between the title's and the null slot at the heap's end, not
/// 2^18 of them (`crate::descriptor::host::HostHeap::new`).
pub(crate) fn dynamic_slots() -> bool {
    switches()[0].1
}

/// `VICTORY_L7_CB_SLOTS`: constant buffer views of the title's heap are not
/// made by the pass over the heap's buffer views (shaders that pick
/// buffers by index read no constant buffer: those are read through
/// tables made afresh), and a constant buffer view's bytes are brought up
/// to date by the draws that read its slot (`buffer_range` knows them).
pub(crate) fn cb_slots() -> bool {
    switches()[1].1
}

/// `VICTORY_L7_VIEW_COPY_LRU`: views' own copies (`VICTORY_L6_VIEW_COPY`)
/// within a budget of bytes (`VICTORY_L7_VIEW_COPY_MB`, 512 by default),
/// the least recently used that nothing else holds let go first; one a
/// slot or a list still holds is never let go (`crate::shadow::exact`).
pub(crate) fn view_copy_lru() -> bool {
    switches()[2].1
}

/// `VICTORY_L7_VIEW_ADDRESS`: a shader that takes the address of a table's
/// view gets it for constant buffer views and views of memory too, not
/// only for buffer views of resources (`crate::draw` `host_word`).
pub(crate) fn view_address() -> bool {
    switches()[3].1
}

/// `VICTORY_L7_TEXTURE_SLOTS`: a slot the title writes with a view of a
/// texture it placed in memory, whose host copy is made already, is made
/// at the write instead of read as null until the texture pass or a draw
/// makes it (`crate::descriptor::host::HostHeap::mirror`).
pub(crate) fn texture_slots() -> bool {
    switches()[4].1
}

/// `VICTORY_L7_VIEW_COPY_MB`: the bytes views' own copies may take.
pub(crate) fn view_copy_budget() -> u64 {
    static B: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *B.get_or_init(|| {
        std::env::var("VICTORY_L7_VIEW_COPY_MB")
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok())
            .unwrap_or(512)
            << 20
    })
}

/// Lot 8: the most views' own copies kept (`VICTORY_L8_VIEW_COPY_MAX`,
/// default 4096: 256 MB at 64 KiB each).
pub(crate) fn view_copy_max() -> usize {
    static M: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *M.get_or_init(|| {
        std::env::var("VICTORY_L8_VIEW_COPY_MAX")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .unwrap_or(4096)
            .max(64)
    })
}

/// Logs the switches, once.
pub(crate) fn announce() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        let s = switches();
        tracing::info!(
            "lot 7 (geometry and textures that come and go) switches, VICTORY_L7_<NAME>=0 turns one off: {}; \
             VICTORY_L7_VIEW_COPY_MB={}",
            s.iter()
                .map(|(n, on)| format!("{n}={}", if *on { "on" } else { "off" }))
                .collect::<Vec<_>>()
                .join(" "),
            view_copy_budget() >> 20
        );
    });
}

/// Counts since the last [`log`].
pub(crate) const CB_LEFT: usize = 0;
pub(crate) const OWN_LET_GO: usize = 1;
pub(crate) const ADDRESS_CONSTANT: usize = 2;
pub(crate) const ADDRESS_PLACED: usize = 3;
pub(crate) const ADDRESS_NONE: usize = 4;
pub(crate) const ADDRESS_OTHER: usize = 5;
pub(crate) const TEXTURE_MADE: usize = 6;
pub(crate) const TEXTURE_LATER: usize = 7;
const COUNTS: usize = 8;
static COUNT: [AtomicU64; COUNTS] = [const { AtomicU64::new(0) }; COUNTS];

/// Gauges: views' own copies alive and their bytes.
pub(crate) static OWN_ALIVE: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

pub(crate) fn add(what: usize, n: u64) {
    COUNT[what].fetch_add(n, Relaxed);
}

/// The counts per frame over `frames`, then zeroed.
pub(crate) fn log(frames: f64) {
    let c: [u64; COUNTS] = std::array::from_fn(|i| COUNT[i].swap(0, Relaxed));
    let f = |i: usize| c[i] as f64 / frames.max(1.0);
    tracing::info!(
        "lot 7 per frame: {:.1} constant buffer views left to the draws that read them (CB_SLOTS); views' own \
         copies: {} alive {} MB, {:.1} let go (VIEW_COPY_LRU); addresses of views taken by shaders: {:.1} of \
         constant buffers, {:.1} of views of memory (VIEW_ADDRESS), {:.1} of slots with no view of ours, {:.1} of \
         other views (both read 0); texture slots written: {:.1} made at the write, their host copy there \
         (TEXTURE_SLOTS), {:.1} left for the texture pass or a draw",
        f(CB_LEFT),
        OWN_ALIVE[0].load(Relaxed),
        OWN_ALIVE[1].load(Relaxed) >> 20,
        f(OWN_LET_GO),
        f(ADDRESS_CONSTANT),
        f(ADDRESS_PLACED),
        f(ADDRESS_NONE),
        f(ADDRESS_OTHER),
        f(TEXTURE_MADE),
        f(TEXTURE_LATER),
    );
}
