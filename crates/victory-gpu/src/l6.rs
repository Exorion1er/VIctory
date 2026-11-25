//! Lot 6, missing geometry: the switches of its behaviour changes (each on
//! by default, `VICTORY_L6_<NAME>=0` turns it off, so that one build can be
//! bisected by environment) and the per-frame counts of the host slots that
//! read nothing.

use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

/// Whether switch `name` (`VICTORY_L6_<name>`) is on: unset or anything but
/// `0`.
fn on(name: &str) -> bool {
    std::env::var_os(format!("VICTORY_L6_{name}")).is_none_or(|v| v != "0")
}

/// The switches, read once: (name, on).
fn switches() -> &'static [(&'static str, bool); 5] {
    static S: std::sync::OnceLock<[(&str, bool); 5]> = std::sync::OnceLock::new();
    S.get_or_init(|| ["HEAP_SYNC", "SUBMIT_SLOTS", "VIEW_COPY", "LATE_ALIAS", "LATE_UNCOPIED"].map(|n| (n, on(n))))
}

/// `VICTORY_L6_HEAP_SYNC`: records the title copies into its shader-visible
/// heaps as bytes (no call) are mirrored at each submission
/// (`crate::descriptor::sync_heaps_for`).
pub(crate) fn heap_sync() -> bool {
    switches()[0].1
}

/// `VICTORY_L6_SUBMIT_SLOTS`: buffer views written into the view heap since
/// the last submission get their host slot (and their bytes) at the
/// submission, not at the next frame's pass over the heap
/// (`crate::draw::submit_slots`).
pub(crate) fn submit_slots() -> bool {
    switches()[1].1
}

/// `VICTORY_L6_VIEW_COPY`: a buffer view that cannot start where its bytes
/// lie in its buffer's host copy (off an element, a raw view not on 16
/// bytes, a constant buffer not on 256) is made over a copy of its own
/// bytes instead of being left null (`crate::shadow::exact`).
pub(crate) fn view_copy() -> bool {
    switches()[2].1
}

/// `VICTORY_L6_LATE_ALIAS` (fork PR #191, fork issue #89): with flights in
/// the air (`VICTORY_LATE_WAIT`, `VICTORY_RELAY`), a submission that
/// refreshes a host copy over title memory that a flight's shaders write
/// through another copy (a window over a buffer, a view's own copy), that
/// a flight's texture read-back or bottom-of-pipe write lands in, waits for
/// that flight first. Without it the refresh took the title's bytes from
/// before the flight's writes were copied back: skinned vertices from an
/// earlier frame or zeros, on whichever submissions the card was behind.
pub(crate) fn late_alias() -> bool {
    switches()[3].1
}

/// `VICTORY_L6_LATE_UNCOPIED` (fork PR #206): with flights in the air, a
/// refresh of a range a flight's shaders may write also takes the pages
/// the copy never took (a copy made since: a new window, a view's own
/// copy), not only those the write-watch journal names; they stayed zeros.
pub(crate) fn late_uncopied() -> bool {
    switches()[4].1
}

/// Logs the switches, once.
pub(crate) fn announce() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        let s = switches();
        tracing::info!(
            "lot 6 (missing geometry) switches, VICTORY_L6_<NAME>=0 turns one off: {}",
            s.iter()
                .map(|(n, on)| format!("{n}={}", if *on { "on" } else { "off" }))
                .collect::<Vec<_>>()
                .join(" ")
        );
    });
}

/// Counts since the last [`log`].
pub(crate) const VIEW_ADDRESS_ZERO: usize = 0;
pub(crate) const OFF_ELEMENT: usize = 1;
pub(crate) const REFUSED: usize = 2;
pub(crate) const CONSTANT_UNALIGNED: usize = 3;
pub(crate) const SYNCED_SLOTS: usize = 4;
pub(crate) const SYNCED_PAGES: usize = 5;
pub(crate) const SUBMIT_MADE: usize = 6;
pub(crate) const OWN_COPIES: usize = 7;
pub(crate) const OWN_REFUSED: usize = 8;
pub(crate) const SYNC_US: usize = 9;
const COUNTS: usize = 10;
static COUNT: [AtomicU64; COUNTS] = [const { AtomicU64::new(0) }; COUNTS];

pub(crate) fn add(what: usize, n: u64) {
    COUNT[what].fetch_add(n, Relaxed);
}

/// The counts per frame over `frames`, then zeroed.
pub(crate) fn log(frames: f64) {
    let c: [u64; COUNTS] = std::array::from_fn(|i| COUNT[i].swap(0, Relaxed));
    let f = |i: usize| c[i] as f64 / frames.max(1.0);
    tracing::info!(
        "lot 6 per frame: {:.1} buffer-view addresses read as 0; host slots left null: {:.1} buffer views off \
         their copy's elements, {:.1} buffer views the host would refuse, {:.1} constant buffers off 256 bytes; \
         heap sync (HEAP_SYNC): {:.1} slots changed without a call in {:.1} written 64 KiB pages, {:.2} ms; \
         {:.1} buffer slots made at submission (SUBMIT_SLOTS); views given a copy of their own (VIEW_COPY): \
         {:.1} made, {:.1} refused (no room)",
        f(VIEW_ADDRESS_ZERO),
        f(OFF_ELEMENT),
        f(REFUSED),
        f(CONSTANT_UNALIGNED),
        f(SYNCED_SLOTS),
        f(SYNCED_PAGES),
        f(SYNC_US) / 1e3,
        f(SUBMIT_MADE),
        f(OWN_COPIES),
        f(OWN_REFUSED),
    );
}
