//! Lot 8, write watch: what our tracking of title memory asks of the
//! system (the kernel's write watch, the guards' protections and queries,
//! reads through the kernel, the guards' faults) counted per frame, and the
//! switches of the changes that ask less (`VICTORY_L8_<NAME>`; `=0` turns
//! one that is on by default off, `=1` turns one that is off by default on).
//!
//! Under Wine every one of these calls takes ntdll's process-wide lock of
//! the address space, which the title's own mappings hold across a call to
//! the wineserver: one call waits behind the other, so the count matters
//! more than the bytes.

use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

/// The switches: name, on by default.
const SWITCHES: &[(&str, bool)] = &[("WW_KNOWN_VIEWS", true), ("WW_ABSENT_TRUST", true)];

/// Whether switch `name` (`VICTORY_L8_<name>`) is on.
fn read(name: &str, default: bool) -> bool {
    match std::env::var_os(format!("VICTORY_L8_{name}")) {
        None => default,
        Some(v) => v != "0",
    }
}

/// The switches, read once.
fn switches() -> &'static [bool] {
    static S: std::sync::OnceLock<Vec<bool>> = std::sync::OnceLock::new();
    S.get_or_init(|| SWITCHES.iter().map(|&(n, d)| read(n, d)).collect())
}

/// Whether the switch called `name` is on.
fn on(name: &str) -> bool {
    SWITCHES
        .iter()
        .position(|&(n, _)| n == name)
        .is_some_and(|i| switches()[i])
}

/// Logs the switches, once.
pub(crate) fn announce() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        let s = switches();
        tracing::info!(
            "lot 8 (write watch) switches, VICTORY_L8_<NAME>=0 or =1: {}",
            SWITCHES
                .iter()
                .zip(s)
                .map(|((n, _), on)| format!("{n}={}", if *on { "on" } else { "off" }))
                .collect::<Vec<_>>()
                .join(" ")
        );
    });
}

/// `VICTORY_L8_WW_KNOWN_VIEWS` (on): no `GetWriteWatch` for a range that
/// begins in a placeholder reservation of the runtime's (`xmem`, where
/// physical pages are mapped as views): such memory is never write-watched,
/// the call fails and the guards answer. It failed for every check of view
/// memory (and twice for a texture there), and a submission's one asking
/// split a run of view ranges in halves until each failed alone, a call
/// each: calls that took the address space's lock for nothing.
pub(crate) fn known_views() -> bool {
    on("WW_KNOWN_VIEWS")
}

/// `VICTORY_L8_WW_ABSENT_TRUST` (on; under Wine only): a block of a
/// placeholder reservation of the runtime's that the guards found unmapped
/// stays so until the runtime maps it, which it tells
/// ([`crate::title_memory_remapped`]): no `VirtualQuery` at every check of
/// a range that holds such blocks, nor for each block of a run that could
/// not be guarded. Under Wine the runtime's sections are committed whole
/// (`xmem`), so a mapped view has no uncommitted pages and a placeholder
/// takes no commit; on Windows a view of a reserved section can be
/// committed without a word, and the query stays.
pub(crate) fn absent_trust() -> bool {
    on("WW_ABSENT_TRUST") && on_wine()
}

/// Whether we run under Wine (its ntdll exports `wine_get_version`).
pub(crate) fn on_wine() -> bool {
    static WINE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *WINE.get_or_init(|| {
        use windows_sys::Win32::System::LibraryLoader::{GetModuleHandleA, GetProcAddress};
        // SAFETY: plain lookups in ntdll, always loaded.
        unsafe {
            let ntdll = GetModuleHandleA(c"ntdll.dll".as_ptr().cast());
            !ntdll.is_null() && GetProcAddress(ntdll, c"wine_get_version".as_ptr().cast()).is_some()
        }
    })
}

/// The calls counted, each with its microseconds.
#[derive(Clone, Copy)]
pub(crate) enum Site {
    /// `GetWriteWatch` that answered.
    WatchOk,
    /// `GetWriteWatch` that failed (memory not write-watched: view memory,
    /// a range across reservations).
    WatchFailed,
    /// The guards' check of a range of view memory (`watch::guard`, its
    /// system calls included).
    Guard,
    /// `VirtualProtect` read-only (arming a guard).
    Protect,
    /// `VirtualProtect` read-write (opening one).
    Open,
    /// `VirtualQuery`.
    Query,
    /// `QueryWorkingSetEx`.
    WorkingSet,
    /// `ReadProcessMemory` (copies of title memory).
    Read,
    /// A guard's fault, the handler's time (its waits included).
    Fault,
}

const SITES: usize = 9;

static CALLS: [AtomicU64; SITES] = [const { AtomicU64::new(0) }; SITES];
static MICROS: [AtomicU64; SITES] = [const { AtomicU64::new(0) }; SITES];

/// Other counts of the line.
#[derive(Clone, Copy)]
pub(crate) enum Count {
    /// Microseconds the title's threads (and the runtime's mapping) waited
    /// for the guards' lock.
    GuardsWaitUs,
    /// Microseconds waited for the write-watch journal's lock.
    JournalWaitUs,
    /// Guard faults on a block that faulted earlier in the same frame.
    FaultsAgain,
    /// `GetWriteWatch` calls not made: the range is view memory
    /// ([`known_views`]).
    WatchesSkipped,
    /// `VirtualQuery` calls not made: blocks known unmapped
    /// ([`absent_trust`]).
    QueriesSkipped,
}

const COUNTS: usize = 5;

static COUNT: [AtomicU64; COUNTS] = [const { AtomicU64::new(0) }; COUNTS];

/// `f`, counted at `site`.
#[inline]
pub(crate) fn timed<T>(site: Site, f: impl FnOnce() -> T) -> T {
    let started = std::time::Instant::now();
    let r = f();
    add_call(site, started.elapsed().as_micros() as u64);
    r
}

/// One call at `site` that took `us`.
#[inline]
pub(crate) fn add_call(site: Site, us: u64) {
    CALLS[site as usize].fetch_add(1, Relaxed);
    MICROS[site as usize].fetch_add(us, Relaxed);
}

/// Adds `n` to `count`.
#[inline]
pub(crate) fn add(count: Count, n: u64) {
    COUNT[count as usize].fetch_add(n, Relaxed);
}

/// `lock()`, its wait counted under `count`.
#[inline]
pub(crate) fn waited<T>(count: Count, lock: impl FnOnce() -> T) -> T {
    let started = std::time::Instant::now();
    let r = lock();
    add(count, started.elapsed().as_micros() as u64);
    r
}

/// The line, over `frames` frames; the counts start again.
pub(crate) fn log(frames: f64) {
    let f = frames.max(1.0);
    let calls: [f64; SITES] = std::array::from_fn(|i| CALLS[i].swap(0, Relaxed) as f64 / f);
    let ms: [f64; SITES] = std::array::from_fn(|i| MICROS[i].swap(0, Relaxed) as f64 / 1e3 / f);
    let c: [f64; COUNTS] = std::array::from_fn(|i| COUNT[i].swap(0, Relaxed) as f64 / f);
    let s = |site: Site| (calls[site as usize], ms[site as usize]);
    let (ok, ok_ms) = s(Site::WatchOk);
    let (failed, failed_ms) = s(Site::WatchFailed);
    let (guard, guard_ms) = s(Site::Guard);
    let (protect, protect_ms) = s(Site::Protect);
    let (open, open_ms) = s(Site::Open);
    let (query, query_ms) = s(Site::Query);
    let (set, set_ms) = s(Site::WorkingSet);
    let (read, read_ms) = s(Site::Read);
    let (fault, fault_ms) = s(Site::Fault);
    tracing::info!(
        "lot 8 per frame: GetWriteWatch {ok:.0} answered {ok_ms:.2} ms, {failed:.0} failed {failed_ms:.2} ms, {:.0} not made \
         (view memory, WW_KNOWN_VIEWS); \
         guard checks {guard:.0} {guard_ms:.2} ms; VirtualProtect {protect:.0} read-only {protect_ms:.2} ms, \
         {open:.0} read-write {open_ms:.2} ms; VirtualQuery {query:.0} {query_ms:.2} ms ({:.0} not made: \
         blocks known unmapped, WW_ABSENT_TRUST); QueryWorkingSetEx \
         {set:.0} {set_ms:.2} ms; ReadProcessMemory {read:.0} {read_ms:.2} ms; guard faults {fault:.0} \
         ({:.0} on a block that faulted earlier in the frame) {fault_ms:.2} ms in the handler; waits for our \
         locks: guards {:.2} ms, write-watch journal {:.2} ms",
        c[Count::WatchesSkipped as usize],
        c[Count::QueriesSkipped as usize],
        c[Count::FaultsAgain as usize],
        c[Count::GuardsWaitUs as usize] / 1e3,
        c[Count::JournalWaitUs as usize] / 1e3,
    );
}
