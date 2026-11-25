//! Lot 7, frame time: the switches of its behaviour changes
//! (`VICTORY_L7_<NAME>`; each on by default, `=0` turns it off, so that one
//! build can be bisected by environment).

/// Whether switch `name` (`VICTORY_L7_<name>`) is on: unset or anything but
/// `0`.
fn on(name: &str) -> bool {
    std::env::var_os(format!("VICTORY_L7_{name}")).is_none_or(|v| v != "0")
}

const NAMES: [&str; 1] = ["PIPELINE_WAIT_ONCE"];

/// The switches, read once: (name, on).
fn switches() -> &'static [(&'static str, bool); 1] {
    static S: std::sync::OnceLock<[(&str, bool); 1]> = std::sync::OnceLock::new();
    S.get_or_init(|| NAMES.map(|n| (n, on(n))))
}

/// `VICTORY_L7_PIPELINE_WAIT_ONCE`: a pipeline is waited for (within the
/// frame's budget, `VICTORY_PIPELINE_WAIT_MS`) in the first frame that asks
/// for it only. One still being made in a later frame is dropped at once
/// for that frame, as it is past the budget. Six compute pipelines never
/// finish on RADV (runs 9-11: translated, bound, never made in 42 minutes):
/// the first dispatch of each frame waited the whole budget for them,
/// 50.1 ms of every frame of run 11.
pub(crate) fn pipeline_wait_once() -> bool {
    switches()[0].1
}

/// Logs the switches, once.
pub(crate) fn announce() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        let s = switches();
        tracing::info!(
            "lot 7 (frame time) switches, VICTORY_L7_<NAME>=0 turns one off: {}",
            s.iter()
                .map(|(n, on)| format!("{n}={}", if *on { "on" } else { "off" }))
                .collect::<Vec<_>>()
                .join(" ")
        );
    });
}
