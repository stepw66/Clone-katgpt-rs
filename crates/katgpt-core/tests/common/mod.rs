//! Shared `CountingAllocator` test/bench infrastructure (Issue 044 T3).
//!
//! Provides a `counting_allocator!()` macro that emits a global allocator
//! tracking alloc/dealloc counts, plus an `alloc_delta` helper. Eliminates
//! ~25 lines of boilerplate per G3/G4 alloc-check test and bench.
//!
//! # Usage
//!
//! In a test file under `tests/`:
//! ```ignore
//! #[path = "common/mod.rs"]
//! mod common;
//! counting_allocator!();
//! ```
//!
//! In a bench file under `benches/`:
//! ```ignore
//! #[path = "../tests/common/mod.rs"]
//! mod common;
//! counting_allocator!();
//! ```
//!
//! After macro invocation, the following names are in scope at the crate
//! root (where the macro was invoked):
//! - `ALLOC_COUNT` — `AtomicUsize` of total allocations
//! - `DEALLOC_COUNT` — `AtomicUsize` of total deallocations
//! - `alloc_delta(f)` — runs `f()` and returns `(result, alloc_count_delta)`
//!
//! Callers that read counters directly (most tests/benches) should add their
//! own `use std::sync::atomic::Ordering;` — the macro does NOT emit `use`
//! statements, to avoid import conflicts at the call site.
//!
//! # Why `#[macro_export]` despite the API pollution concern?
//!
//! `macro_rules!` macros without `#[macro_export]` are NOT path-accessible
//! from sibling modules — only from descendants of the defining module. Since
//! test/bench files invoke the macro from the CRATE ROOT (not from inside
//! `mod common`), `#[macro_export]` is required to make the macro visible.
//! The export goes to the consuming test/bench crate's root (each test/bench
//! is its own crate), NOT to `katgpt_core`'s public API — `tests/common/mod.rs`
//! is not compiled into the `katgpt_core` library crate.

/// Install a global `CountingAllocator` that counts alloc/dealloc calls.
///
/// Emits, at the call site:
/// - `struct CountingAllocator;`
/// - `static ALLOC_COUNT: AtomicUsize`
/// - `static DEALLOC_COUNT: AtomicUsize`
/// - `unsafe impl GlobalAlloc for CountingAllocator`
/// - `#[global_allocator] static A: CountingAllocator`
/// - `fn alloc_delta<R>(f: impl FnOnce() -> R) -> (R, usize)`
///
/// All paths are fully qualified to avoid import conflicts. Callers needing
/// `Ordering` for direct counter reads should add their own
/// `use std::sync::atomic::Ordering;`.
#[macro_export]
macro_rules! counting_allocator {
    () => {
        struct CountingAllocator;

        static ALLOC_COUNT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        static DEALLOC_COUNT: std::sync::atomic::AtomicUsize =
            std::sync::atomic::AtomicUsize::new(0);

        unsafe impl std::alloc::GlobalAlloc for CountingAllocator {
            unsafe fn alloc(&self, layout: std::alloc::Layout) -> *mut u8 {
                ALLOC_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                unsafe { std::alloc::System.alloc(layout) }
            }
            unsafe fn dealloc(&self, ptr: *mut u8, layout: std::alloc::Layout) {
                DEALLOC_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                unsafe { std::alloc::System.dealloc(ptr, layout) }
            }
        }

        #[global_allocator]
        static A: CountingAllocator = CountingAllocator;

        #[inline]
        #[allow(dead_code)]
        fn alloc_delta<R>(f: impl FnOnce() -> R) -> (R, usize) {
            let before = ALLOC_COUNT.load(std::sync::atomic::Ordering::Relaxed);
            let r = f();
            let after = ALLOC_COUNT.load(std::sync::atomic::Ordering::Relaxed);
            (r, after - before)
        }
    };
}
