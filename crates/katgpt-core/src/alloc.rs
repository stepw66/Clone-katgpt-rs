//! Debug-only allocation tracking.
//!
//! Counters are **per-thread** (thread-local `Cell`), not process-global. This
//! lets parallel tests measure allocation-free hot paths without bleeding
//! sibling-test allocations into each other's counts. Every existing caller
//! follows `reset → measure-on-calling-thread → get`, which is exactly the
//! per-thread semantic; switching from global atomics to thread-local `Cell`
//! is both more correct (natural attribution) and faster on the hot path
//! (no `lock cmpxchg`, no cross-core cache-line bouncing).
//!
//! `const` initialization in `thread_local!` avoids lazy-init allocation on
//! the fast path — the TLS slot holds the `Cell` directly.
//!
//! **Whole-module `debug_assertions` gate:** every item below — the
//! `TrackingAllocator`, the `THREAD_ALLOC` slot, the `AllocStats` record, and
//! the `reset`/`get` accessors — exists *only* under `debug_assertions`. In a
//! release build the entire tracking machinery compiles away to nothing: the
//! binary installs the plain `System` allocator (see `lib.rs`'s
//! `#[cfg(all(test, debug_assertions))]` `TEST_GLOBAL_ALLOC`), there are no
//! counters, and there is nothing to test. The `tests` module is therefore
//! gated on `cfg(all(test, debug_assertions))` so a `--release` test build
//! (where `cfg(test)` is on but `debug_assertions` is off) does not reference
//! absent symbols.

#[cfg(debug_assertions)]
use std::alloc::{GlobalAlloc, Layout, System};
#[cfg(debug_assertions)]
use std::cell::Cell;

/// Aggregate per-thread allocation stats (count + bytes). `Copy` so it can
/// live in a `Cell` (single load + store per `alloc`, no `RefCell` overhead).
#[cfg(debug_assertions)]
#[derive(Clone, Copy)]
struct AllocStats {
    count: usize,
    bytes: usize,
}

#[cfg(debug_assertions)]
impl AllocStats {
    /// `const` constructor so the `thread_local!` initializer is const-evaluable.
    const ZERO: Self = Self { count: 0, bytes: 0 };
}

#[cfg(debug_assertions)]
thread_local! {
    /// Single TLS key for both counters — one TLS address computation per
    /// `alloc`, not two.
    static THREAD_ALLOC: Cell<AllocStats> = const { Cell::new(AllocStats::ZERO) };
}

/// Debug-only allocator wrapper that tracks allocation count and bytes on the
/// **calling thread**. Install via `#[global_allocator]` in the binary crate.
#[cfg(debug_assertions)]
pub struct TrackingAllocator;

#[cfg(debug_assertions)]
unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        THREAD_ALLOC.with(|cell| {
            let mut s = cell.get();
            s.count = s.count.wrapping_add(1);
            s.bytes = s.bytes.wrapping_add(layout.size());
            cell.set(s);
        });
        // Safety: delegated to the system allocator, layout is valid.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // Safety: delegated to the system allocator, ptr+layout are valid.
        unsafe { System.dealloc(ptr, layout) }
    }
}

/// Reset the **calling thread's** allocation counters to zero. Does not affect
/// other threads' counters — each thread's `Cell` is independent.
#[cfg(debug_assertions)]
pub fn reset_alloc_stats() {
    THREAD_ALLOC.with(|cell| cell.set(AllocStats::ZERO));
}

/// Get the **calling thread's** allocation stats as `(count, total_bytes)`.
/// Returns only allocations performed on the current thread since the last
/// [`reset_alloc_stats`] on this thread.
#[cfg(debug_assertions)]
pub fn get_alloc_stats() -> (usize, usize) {
    THREAD_ALLOC.with(|cell| {
        let s = cell.get();
        (s.count, s.bytes)
    })
}

// See module docs: the tests exercise accessors that only exist under
// `debug_assertions`, so the module must be gated to match. A `--release`
// test build has `cfg(test)` on but `debug_assertions` off — without this
// gate the tests would reference absent symbols and fail to compile.
#[cfg(all(test, debug_assertions))]
mod tests {
    use super::*;

    // No serialization mutex needed: each test thread's counters are isolated
    // by the thread-local `Cell`. Tests run fully parallel without interference.

    #[test]
    fn test_reset_clears_stats() {
        // Reset and immediately read on THIS thread. With thread-local
        // counters, concurrent tests on other threads cannot inflate our
        // count. Between reset and read this thread performs no allocations,
        // so count should be exactly 0. A small tolerance is kept only to
        // absorb any runtime bookkeeping the test harness itself might do on
        // this thread before reaching the assertion.
        reset_alloc_stats();
        let (count, bytes) = get_alloc_stats();
        assert!(
            count <= 5,
            "count should be near-zero after reset, got {count}"
        );
        assert!(
            bytes <= 4096,
            "bytes should be near-zero after reset, got {bytes}"
        );
    }

    #[test]
    fn test_alloc_increments_count() {
        reset_alloc_stats();
        let _v: Vec<u8> = vec![0u8; 1024];
        let (count, bytes) = get_alloc_stats();
        assert!(count > 0, "at least one allocation should have occurred");
        assert!(bytes >= 1024, "bytes should be at least 1024, got {bytes}");
    }

    #[test]
    fn test_multiple_allocs_accumulate() {
        reset_alloc_stats();
        let _v1: Vec<u8> = vec![0u8; 64];
        let _v2: Vec<u8> = vec![0u8; 128];
        let (count, bytes) = get_alloc_stats();
        assert!(count >= 2, "at least two allocations, got {count}");
        assert!(bytes >= 192, "bytes should be at least 192, got {bytes}");
    }

    #[test]
    fn test_string_allocation() {
        reset_alloc_stats();
        let _s = String::from("hello world test allocation");
        let (count, bytes) = get_alloc_stats();
        assert!(count > 0, "string allocation should increment count");
        assert!(bytes > 0, "string allocation should increment bytes");
    }

    #[test]
    fn test_box_allocation() {
        reset_alloc_stats();
        let _b = Box::new(42u64);
        let (count, bytes) = get_alloc_stats();
        assert!(count > 0, "box allocation should increment count");
        assert!(
            bytes >= 8,
            "box allocation should account for u64, got {bytes}"
        );
    }

    /// Thread isolation: allocations on another thread are not visible on
    /// this thread. This is the property G7 (and every other alloc-audit
    /// gate) relies on for parallel-safe measurement.
    #[test]
    fn test_thread_isolation() {
        // The worker thread allocates 4 KiB on its own thread and reports its
        // own observed byte count via the channel.
        const WORKER_ALLOC_BYTES: usize = 4096;
        let (tx, rx) = std::sync::mpsc::channel();
        let handle = std::thread::spawn(move || {
            reset_alloc_stats();
            let _v: Vec<u8> = vec![0u8; WORKER_ALLOC_BYTES];
            let (_count, bytes) = get_alloc_stats();
            let _ = tx.send(bytes);
        });
        // Reset on THIS thread AFTER spawning (thread spawn itself allocates
        // bookkeeping on the spawning thread — that's legitimate runtime
        // overhead, not the property under test).
        reset_alloc_stats();
        let worker_bytes = rx.recv().expect("worker should report");
        handle.join().expect("worker thread panicked");
        let (_main_count, main_bytes) = get_alloc_stats();
        // The worker saw its own ~4 KiB allocation (modulo Vec bookkeeping).
        assert!(
            worker_bytes >= WORKER_ALLOC_BYTES,
            "worker thread should have seen its own {}-byte allocation, got {}",
            WORKER_ALLOC_BYTES,
            worker_bytes
        );
        // The main thread must NOT see the worker's allocation. Some runtime
        // bookkeeping from `recv`/`join` may allocate on this thread, but it
        // is tiny (hundreds of bytes) — nowhere near 4 KiB. The defining
        // property: the worker's 4 KiB does not leak into our counter.
        assert!(
            main_bytes < WORKER_ALLOC_BYTES,
            "main thread should not see the worker's {WORKER_ALLOC_BYTES}-byte \
             allocation, but observed {main_bytes} bytes on this thread — \
             thread-local isolation is broken"
        );
    }
}
