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
const SWITCHES: &[(&str, bool)] = &[("WW_KNOWN_VIEWS", true), ("WW_ABSENT_TRUST", true), ("WW_BY_RESERVATION", false), ("WW_PROTECT_RUNS", true), ("GUARD_REARM_FRAME", false)];

/// `VICTORY_L8_WW_CHUNK_MB` (0 = off): the most one `GetWriteWatch` call
/// walks of a write-watched reservation of the runtime's; a longer asking is
/// made in pieces. Under GE-Proton 11 a call walks its whole range
/// (`PAGEMAP_SCAN`, about 0.33 us per MiB: 330 us for a 1 GiB heap) at a
/// stretch, holding the address space's locks the title's mappings need;
/// in pieces the title can get in between. The micro-benchmark saw neither
/// gain nor loss at 64 (the title's remaps cost the same), a loss at 16:
/// off until a run's "lot 8" line shows long answered calls.
pub(crate) fn chunk() -> usize {
    static MB: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *MB.get_or_init(|| {
        std::env::var("VICTORY_L8_WW_CHUNK_MB").ok().and_then(|v| v.parse().ok()).unwrap_or(0)
    }) << 20
}

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
            "lot 8 (write watch) switches, VICTORY_L8_<NAME>=0 or =1: {} WW_CHUNK_MB={}",
            SWITCHES
                .iter()
                .zip(s)
                .map(|((n, _), on)| format!("{n}={}", if *on { "on" } else { "off" }))
                .collect::<Vec<_>>()
                .join(" "),
            chunk() >> 20
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

/// `VICTORY_L8_WW_BY_RESERVATION` (off): a submission's write-watch asking
/// covers all its ranges in one write-watched reservation of the runtime's
/// with one call (up to 1 GiB), however far apart they lie, instead of one
/// call per group of ranges within 8 MiB of each other. Fewer calls, each
/// waiting for the address space's lock once; but under GE-Proton 11's
/// kernel write watch (`PAGEMAP_SCAN`) a call walks the whole range, about
/// 0.33 us per MiB (330 us for 1 GiB against 2 us for 4 MiB, `bench`), so
/// it pays only while the waits are long.
pub(crate) fn by_reservation() -> bool {
    on("WW_BY_RESERVATION")
}

/// `VICTORY_L8_WW_PROTECT_RUNS` (on): the guards change protections a run
/// of blocks at a time, cut where the runtime's views begin and end. A run
/// across views was asked first and refused (a protection change stays in
/// one view), then done block by block; the other views of guarded pages
/// were armed, and every opening done, one block per call.
pub(crate) fn protect_runs() -> bool {
    on("WW_PROTECT_RUNS")
}

/// `VICTORY_L8_GUARD_REARM_FRAME` (off): a guarded block the title writes
/// again after its guard was armed again in the same frame is left open
/// until the next frame: every check in the meantime reports it written
/// (and its copy takes it again), with no more faults nor protection
/// changes for it in that frame. Worth it only where the line's "faulted
/// earlier in the frame" count is high: each fault under Wine is a signal,
/// an exception dispatch and two protection changes on the title's thread,
/// against a 64 KiB copy per check of the block.
pub(crate) fn rearm_frame() -> bool {
    on("GUARD_REARM_FRAME")
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
    /// Guards left open until the next frame ([`rearm_frame`]).
    RearmDeferred,
    /// AnyXboxX lot 9: blocks a check left open because a kernel write of
    /// the runtime's was under way there (`crate::watch::kernel_write`).
    HeldOpen,
    /// AnyXboxX lot 9 fix (`VICTORY_L9_GUARD_TITLE_VIEWS`): blocks a check
    /// left writable because no title view shows them
    /// (`crate::watch::protect_here`).
    NotTitle,
    /// AnyXboxX lot 9 fix: write faults on the thread that held the guards'
    /// lock, opened without it (`crate::watch::reentrant`).
    Reentrant,
}

const COUNTS: usize = 9;

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
    tracing::info!("{}", line(frames));
}

/// [`log`]'s text.
pub(crate) fn line(frames: f64) -> String {
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
    format!(
        "lot 8 per frame: GetWriteWatch {ok:.0} answered {ok_ms:.2} ms, {failed:.0} failed {failed_ms:.2} ms, {:.0} not made \
         (view memory, WW_KNOWN_VIEWS); \
         guard checks {guard:.0} {guard_ms:.2} ms; VirtualProtect {protect:.0} read-only {protect_ms:.2} ms, \
         {open:.0} read-write {open_ms:.2} ms; VirtualQuery {query:.0} {query_ms:.2} ms ({:.0} not made: \
         blocks known unmapped, WW_ABSENT_TRUST); QueryWorkingSetEx \
         {set:.0} {set_ms:.2} ms; ReadProcessMemory {read:.0} {read_ms:.2} ms; guard faults {fault:.0} \
         ({:.0} on a block that faulted earlier in the frame) {fault_ms:.2} ms in the handler, {:.0} guards \
         left open to the next frame (GUARD_REARM_FRAME), {:.1} left open under a kernel write (L9_DSTORAGE_GUARDS), {:.0} blocks no title view shows left \
         writable (copied whole; L9_GUARD_TITLE_VIEWS), {:.2} write faults under the guards' lock; waits for our \
         locks: guards {:.2} ms, write-watch journal {:.2} ms",
        c[Count::WatchesSkipped as usize],
        c[Count::QueriesSkipped as usize],
        c[Count::FaultsAgain as usize],
        c[Count::RearmDeferred as usize],
        c[Count::HeldOpen as usize],
        c[Count::NotTitle as usize],
        c[Count::Reentrant as usize],
        c[Count::GuardsWaitUs as usize] / 1e3,
        c[Count::JournalWaitUs as usize] / 1e3,
    )
}

/// Micro-benchmark of the old and new ways (ignored; run alone, once per
/// switch setting, under the Wine to be measured):
/// `victory_gpu-*.exe l8ww::bench --ignored --nocapture --test-threads=1`.
/// A synthetic layout like the title's world: `L8_BENCH_VIEWS` (20,000)
/// views of 64 KiB in a placeholder, a write-watched reservation of 1 GiB
/// with 3,000 windows of 4 MiB over it, 1,000 windows of 1 MiB over the
/// views; a thread writing title memory and one remapping views as the
/// runtime does, while the main thread runs 48 submissions a frame of
/// 1,000 refreshes each through the journal and the guards.
#[cfg(test)]
mod bench {
    use crate::watch;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::Relaxed};
    use std::time::{Duration, Instant};
    use windows_sys::Win32::System::Memory::{
        CreateFileMappingW, MEM_COMMIT, MEM_PRESERVE_PLACEHOLDER, MEM_RELEASE, MEM_REPLACE_PLACEHOLDER,
        MEM_RESERVE, MEM_RESERVE_PLACEHOLDER, MEMORY_BASIC_INFORMATION, MEMORY_MAPPED_VIEW_ADDRESS,
        MapViewOfFile3, PAGE_NOACCESS, PAGE_READONLY, PAGE_READWRITE, UnmapViewOfFile2, VirtualAlloc,
        VirtualAlloc2, VirtualFree, VirtualProtect, VirtualQuery,
    };
    use windows_sys::Win32::System::Threading::GetCurrentProcess;

    const K64: usize = 64 << 10;
    const MEM_WRITE_WATCH: u32 = 0x20_0000;
    const WATCHED: usize = 1 << 30;
    const WINDOW: usize = 4 << 20;
    const WATCHED_WINDOWS: usize = 3000;
    const VIEW_WINDOW: usize = 1 << 20;
    const VIEW_WINDOWS: usize = 1000;
    const SUBMISSIONS: usize = 48;
    const REFRESHES: usize = 1000;
    const TEXTURES: usize = 20;
    const HOT_PAGES: usize = 4096;

    fn env(name: &str, default: usize) -> usize {
        std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
    }

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> usize {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0 as usize
        }
        fn below(&mut self, n: usize) -> usize {
            self.next() % n
        }
    }

    struct Layout {
        watched: usize,
        views: usize,
        count: usize,
        section: usize,
    }

    fn map(l: &Layout, i: usize) {
        let at = l.views + i * K64;
        // SAFETY: a placeholder block of ours, the section's block i.
        let view = unsafe {
            MapViewOfFile3(
                l.section as _,
                GetCurrentProcess(),
                at as *const _,
                (i * K64) as u64,
                K64,
                MEM_REPLACE_PLACEHOLDER,
                PAGE_READWRITE,
                std::ptr::null_mut(),
                0,
            )
        };
        assert_eq!(view.Value as usize, at, "map view {i}");
        crate::title_memory_remapped(at, K64);
        crate::inplace::title_memory_view(at, K64, l.section, (i * K64) as u64);
    }

    fn unmap(l: &Layout, i: usize) {
        let at = l.views + i * K64;
        crate::title_memory_remapped(at, K64);
        // SAFETY: a view of ours, back to a placeholder.
        unsafe {
            UnmapViewOfFile2(
                GetCurrentProcess(),
                MEMORY_MAPPED_VIEW_ADDRESS { Value: at as *mut _ },
                MEM_PRESERVE_PLACEHOLDER,
            )
        };
    }

    fn layout() -> Layout {
        let count = env("L8_BENCH_VIEWS", 20_000);
        let started = Instant::now();
        // SAFETY: fresh reservations and section, kept for the process.
        let (watched, views, section) = unsafe {
            let watched = VirtualAlloc(
                std::ptr::null(),
                WATCHED,
                MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
                PAGE_READWRITE,
            ) as usize;
            assert_ne!(watched, 0);
            let views = VirtualAlloc2(
                GetCurrentProcess(),
                std::ptr::null(),
                count * K64,
                MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
                PAGE_NOACCESS,
                std::ptr::null_mut(),
                0,
            ) as usize;
            assert_ne!(views, 0);
            let bytes = (count * K64) as u64;
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                (bytes >> 32) as u32,
                bytes as u32,
                std::ptr::null(),
            ) as usize;
            assert_ne!(section, 0);
            for i in 0..count - 1 {
                VirtualFree((views + i * K64) as *mut _, K64, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER);
            }
            (watched, views, section)
        };
        crate::title_memory_reserved(watched, WATCHED, true, false);
        crate::title_memory_reserved(views, count * K64, false, true);
        let l = Layout { watched, views, count, section };
        for i in 0..count {
            map(&l, i);
        }
        eprintln!("layout: {count} views mapped in {:.1} s", started.elapsed().as_secs_f64());
        l
    }

    /// Microseconds per call of `f`, over `n` calls: mean and the worst.
    fn per_call(n: usize, mut f: impl FnMut(usize)) -> (f64, f64) {
        let mut worst = 0f64;
        let all = Instant::now();
        for i in 0..n {
            let one = Instant::now();
            f(i);
            worst = worst.max(one.elapsed().as_secs_f64() * 1e6);
        }
        (all.elapsed().as_secs_f64() * 1e6 / n as f64, worst)
    }

    fn raw(l: &Layout, rng: &mut Rng, what: &str) {
        use windows_sys::Win32::System::Diagnostics::Debug::ReadProcessMemory;
        use windows_sys::Win32::System::Memory::GetWriteWatch;
        let mut found = vec![0usize; WATCHED / 4096];
        let mut gww = |at: usize, len: usize| {
            let (mut count, mut granularity) = (len / 4096, 0u32);
            // SAFETY: the answer buffer holds `count` entries.
            unsafe { GetWriteWatch(1, at as *const _, len, found.as_mut_ptr().cast(), &mut count, &mut granularity) }
        };
        let n = 2000;
        let w4 = per_call(n, |_| {
            let at = l.watched + rng.below(WATCHED / WINDOW) * WINDOW;
            gww(at, WINDOW);
        });
        let w1g = per_call(50, |_| {
            gww(l.watched, WATCHED);
        });
        let mut rng2 = Rng(rng.next() as u64 | 1);
        let failed = per_call(n, |_| {
            let at = l.views + rng2.below(l.count) * K64;
            gww(at, K64);
        });
        let protect = per_call(n, |_| {
            let at = l.views + rng2.below(l.count) * K64;
            let mut old = 0;
            // SAFETY: a view of ours; back to read-write below.
            unsafe {
                VirtualProtect(at as *const _, K64, PAGE_READONLY, &mut old);
                VirtualProtect(at as *const _, K64, PAGE_READWRITE, &mut old);
            }
        });
        let query = per_call(n, |_| {
            let at = l.views + rng2.below(l.count) * K64;
            // SAFETY: plain data.
            let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
            // SAFETY: a valid out pointer.
            unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) };
        });
        let mut buf = vec![0u8; K64];
        let read = per_call(n, |_| {
            let at = l.views + rng2.below(l.count) * K64;
            let mut done = 0;
            // SAFETY: our buffer of K64 bytes.
            unsafe { ReadProcessMemory(GetCurrentProcess(), at as *const _, buf.as_mut_ptr().cast(), K64, &mut done) };
        });
        let copy = per_call(n, |_| {
            let at = l.views + rng2.below(l.count) * K64;
            let _mapped = watch::title_memory_reading();
            // SAFETY: a mapped view (no remapping under the read lock).
            unsafe { std::ptr::copy_nonoverlapping(at as *const u8, buf.as_mut_ptr(), K64) };
        });
        eprintln!(
            "raw calls {what}, us per call (mean / worst): GetWriteWatch+reset 4 MiB {:.1}/{:.0}, 1 GiB {:.1}/{:.0}, \
             failing (view) {:.1}/{:.0}; VirtualProtect read-only + back (64 KiB view) {:.1}/{:.0}; VirtualQuery \
             {:.1}/{:.0}; ReadProcessMemory 64 KiB {:.1}/{:.0}; memcpy 64 KiB {:.1}/{:.0}",
            w4.0, w4.1, w1g.0, w1g.1, failed.0, failed.1, protect.0, protect.1, query.0, query.1, read.0, read.1,
            copy.0, copy.1
        );
    }

    /// The title: a thread writing memory (hot pages of the watched
    /// reservation, blocks of the views) and one remapping views.
    fn title(l: &Arc<Layout>, stop: &Arc<AtomicBool>, remaps: &Arc<[AtomicU64; 2]>) -> Vec<std::thread::JoinHandle<()>> {
        let writer = {
            let (l, stop) = (l.clone(), stop.clone());
            std::thread::spawn(move || {
                let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
                let hot: Vec<usize> = (0..HOT_PAGES).map(|_| rng.below(WATCHED / 4096)).collect();
                while !stop.load(Relaxed) {
                    for _ in 0..20 {
                        let page = hot[rng.below(HOT_PAGES)];
                        // SAFETY: committed memory of ours.
                        unsafe { ((l.watched + page * 4096) as *mut u8).write_volatile(1) };
                    }
                    {
                        let _mapped = watch::title_memory_reading();
                        let at = l.views + rng.below(l.count) * K64 + 64;
                        // SAFETY: a mapped view (no remapping under the
                        // read lock); a guard's fault is taken.
                        unsafe { (at as *mut u8).write_volatile(1) };
                    }
                    std::thread::sleep(Duration::from_micros(500));
                }
            })
        };
        let remapper = {
            let (l, stop, remaps) = (l.clone(), stop.clone(), remaps.clone());
            std::thread::spawn(move || {
                let mut rng = Rng(0x2545_f491_4f6c_dd1d);
                while !stop.load(Relaxed) {
                    let i = rng.below(l.count);
                    let started = Instant::now();
                    {
                        let _changing = watch::title_memory_remapping();
                        unmap(&l, i);
                        map(&l, i);
                    }
                    remaps[0].fetch_add(1, Relaxed);
                    remaps[1].fetch_add(started.elapsed().as_micros() as u64, Relaxed);
                    std::thread::sleep(Duration::from_millis(1));
                }
            })
        };
        vec![writer, remapper]
    }

    /// One frame of submissions; its microseconds.
    fn frame(l: &Layout, rng: &mut Rng, windows: &[(usize, usize)]) -> u64 {
        let started = Instant::now();
        for _ in 0..SUBMISSIONS {
            let picks: Vec<(usize, usize)> = (0..REFRESHES).map(|_| windows[rng.below(windows.len())]).collect();
            let mut spans = picks.clone();
            spans.sort_unstable();
            spans.dedup();
            watch::ask_once(&spans);
            for &(from, to) in &picks {
                std::hint::black_box(watch::writes(from, to, true));
            }
            for _ in 0..TEXTURES {
                let at = l.views + rng.below(l.count - 16) * K64;
                if watch::writes(at, at + VIEW_WINDOW, false).is_none() {
                    std::hint::black_box(watch::writes(at, at + VIEW_WINDOW, true));
                }
            }
            watch::ask_once_end();
        }
        started.elapsed().as_micros() as u64
    }

    #[test]
    #[ignore]
    fn bench() {
        super::announce();
        eprintln!(
            "switches: {}",
            super::SWITCHES
                .iter()
                .zip(super::switches())
                .map(|((n, _), on)| format!("{n}={}", if *on { "on" } else { "off" }))
                .collect::<Vec<_>>()
                .join(" ")
                + &format!(" WW_CHUNK_MB={}", super::chunk() >> 20)
        );
        let l = Arc::new(layout());
        let mut rng = Rng(0x1234_5678_9abc_def1);
        raw(&l, &mut rng, "(quiet)");
        let mut windows: Vec<(usize, usize)> = (0..WATCHED_WINDOWS)
            .map(|_| {
                let at = l.watched + rng.below((WATCHED - WINDOW) / K64) * K64;
                (at, at + WINDOW)
            })
            .collect();
        windows.extend((0..VIEW_WINDOWS).map(|_| {
            let at = l.views + rng.below(l.count - VIEW_WINDOW / K64) * K64;
            (at, at + VIEW_WINDOW)
        }));
        let stop = Arc::new(AtomicBool::new(false));
        let remaps = Arc::new([AtomicU64::new(0), AtomicU64::new(0)]);
        let threads = title(&l, &stop, &remaps);
        raw(&l, &mut rng, "(title writing and remapping)");
        // Warm: every window taken once.
        frame(&l, &mut rng, &windows);
        let _ = super::line(1.0);
        let (r0, r1) = (remaps[0].load(Relaxed), remaps[1].load(Relaxed));
        let frames = env("L8_BENCH_FRAMES", 10);
        // `L8_BENCH_IDLE_MS`: the main thread sleeps instead (the title's
        // remaps alone).
        let idle = env("L8_BENCH_IDLE_MS", 0);
        let mut times: Vec<u64> = (0..frames)
            .map(|_| {
                if idle > 0 {
                    std::thread::sleep(Duration::from_millis(idle as u64));
                    return idle as u64 * 1000;
                }
                frame(&l, &mut rng, &windows)
            })
            .collect();
        times.sort_unstable();
        let (n, us) = (remaps[0].load(Relaxed) - r0, remaps[1].load(Relaxed) - r1);
        eprintln!(
            "frames: median {:.1} ms, best {:.1} ms, worst {:.1} ms of tracking per frame ({SUBMISSIONS} submissions of \
             {REFRESHES} refreshes + {TEXTURES} texture checks); the title's remaps meanwhile: {n}, {:.0} us each",
            times[frames / 2] as f64 / 1e3,
            times[0] as f64 / 1e3,
            times[frames - 1] as f64 / 1e3,
            us as f64 / n.max(1) as f64
        );
        eprintln!("{}", super::line(frames as f64));
        stop.store(true, Relaxed);
        for t in threads {
            t.join().unwrap();
        }
    }
}
