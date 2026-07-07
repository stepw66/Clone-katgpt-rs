//! HOLA Hippocampal Exact KV Cache GOAT gate G2 latency benchmark (Plan 395).
//!
//! Validates:
//! - `bench_observe` — HippocampalCache<256, 64>, 10k observes. Target: ≤ 100 ns/observe.
//! - `bench_read` — single read_cache_into call. Target: ≤ 1 µs/read at W=64, D=256.
//! - `bench_observe_micro` — HippocampalCache<8, 16> (game micro config). Target: ≤ 30 ns/observe.
//! - `bench_heap_vs_sorted` — T2.3: heap vs sorted-vec eviction comparison.
//!
//! # Run
//!
//! ```bash
//! CARGO_TARGET_DIR=/tmp/hippocampal_cache_goat cargo bench --bench hippocampal_cache_goat --features hippocampal_cache
//! ```

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use katgpt_core::{HippocampalCache, SortedSlotCache};

fn bench_observe(c: &mut Criterion) {
    let mut group = c.benchmark_group("hippocampal_cache/observe");

    // W=64, D=256 (paper-scale).
    group.bench_function("heap_w64_d256", |b| {
        let mut cache: HippocampalCache<256, 64> = HippocampalCache::new_with_ones_gamma();
        let mut i = 0u32;
        b.iter(|| {
            let k = [black_box(i as f32) / 1000.0; 256];
            let v = [black_box(i as f32) / 2000.0; 256];
            cache.observe(black_box(&k), black_box(&v), 0.5, 0.5);
            i = i.wrapping_add(1);
        });
    });

    // W=16, D=8 (micro config).
    group.bench_function("heap_w16_d8_micro", |b| {
        let mut cache: HippocampalCache<8, 16> = HippocampalCache::new_with_ones_gamma();
        let mut i = 0u32;
        b.iter(|| {
            let k = [black_box(i as f32) / 1000.0; 8];
            let v = [black_box(i as f32) / 2000.0; 8];
            cache.observe(black_box(&k), black_box(&v), 0.5, 0.5);
            i = i.wrapping_add(1);
        });
    });

    group.finish();
}

fn bench_read(c: &mut Criterion) {
    let mut group = c.benchmark_group("hippocampal_cache/read");

    // W=64, D=256, softmax read.
    group.bench_function("softmax_w64_d256", |b| {
        let mut cache: HippocampalCache<256, 64> = HippocampalCache::new_with_ones_gamma();
        // Fill cache.
        for i in 0..64 {
            let k = [i as f32 / 100.0; 256];
            let v = [i as f32 / 200.0; 256];
            cache.observe(&k, &v, 0.9, 0.9);
        }
        let q = [0.5f32; 256];
        let gamma = [1.0f32; 256];
        let mut out = [0.0f32; 256];
        b.iter(|| {
            cache.read_cache_into(
                black_box(&q),
                black_box(&gamma),
                black_box(&[]),
                black_box(&mut out),
            );
        });
    });

    // W=16, D=8 (micro config), softmax read.
    group.bench_function("softmax_w16_d8_micro", |b| {
        let mut cache: HippocampalCache<8, 16> = HippocampalCache::new_with_ones_gamma();
        for i in 0..16 {
            let k = [i as f32 / 10.0; 8];
            let v = [i as f32 / 20.0; 8];
            cache.observe(&k, &v, 0.9, 0.9);
        }
        let q = [0.5f32; 8];
        let gamma = [1.0f32; 8];
        let mut out = [0.0f32; 8];
        b.iter(|| {
            cache.read_cache_into(
                black_box(&q),
                black_box(&gamma),
                black_box(&[]),
                black_box(&mut out),
            );
        });
    });

    // W=64, D=256, softmax fast read (pre-normalized keys).
    group.bench_function("softmax_fast_w64_d256", |b| {
        let mut cache: HippocampalCache<256, 64> = HippocampalCache::new_with_ones_gamma();
        for i in 0..64 {
            let k = [i as f32 / 100.0; 256];
            let v = [i as f32 / 200.0; 256];
            cache.observe(&k, &v, 0.9, 0.9);
        }
        let q = [0.5f32; 256];
        let mut out = [0.0f32; 256];
        b.iter(|| {
            cache.read_cache_into_fast(black_box(&q), black_box(&[]), black_box(&mut out));
        });
    });

    // W=64, D=256, sigmoid-gated read (for comparison).
    group.bench_function("sigmoid_w64_d256", |b| {
        let mut cache: HippocampalCache<256, 64> = HippocampalCache::new_with_ones_gamma();
        for i in 0..64 {
            let k = [i as f32 / 100.0; 256];
            let v = [i as f32 / 200.0; 256];
            cache.observe(&k, &v, 0.9, 0.9);
        }
        let q = [0.5f32; 256];
        let gamma = [1.0f32; 256];
        let mut out = [0.0f32; 256];
        b.iter(|| {
            cache.read_cache_into_sigmoid(
                black_box(&q),
                black_box(&gamma),
                black_box(&[]),
                black_box(&mut out),
            );
        });
    });

    group.finish();
}

/// T2.3: heap vs sorted-vec eviction. Both at W=64.
fn bench_heap_vs_sorted(c: &mut Criterion) {
    let mut group = c.benchmark_group("hippocampal_cache/t2_3_heap_vs_sorted");

    group.bench_function("sorted_w64", |b| {
        let mut cache: SortedSlotCache<64, 64> = SortedSlotCache::new();
        let mut i = 0u32;
        b.iter(|| {
            let k = [black_box(i as f32) / 1000.0; 64];
            let v = [black_box(i as f32) / 2000.0; 64];
            cache.observe(black_box(&k), black_box(&v), 0.5, 0.5);
            i = i.wrapping_add(1);
        });
    });

    // Same config, heap-backed (already in bench_observe, but included here
    // for direct side-by-side comparison in the same group).
    group.bench_function("heap_w64", |b| {
        let mut cache: HippocampalCache<64, 64> = HippocampalCache::new_with_ones_gamma();
        let mut i = 0u32;
        b.iter(|| {
            let k = [black_box(i as f32) / 1000.0; 64];
            let v = [black_box(i as f32) / 2000.0; 64];
            cache.observe(black_box(&k), black_box(&v), 0.5, 0.5);
            i = i.wrapping_add(1);
        });
    });

    // Sorted at W=16 (micro config).
    group.bench_function("sorted_w16", |b| {
        let mut cache: SortedSlotCache<8, 16> = SortedSlotCache::new();
        let mut i = 0u32;
        b.iter(|| {
            let k = [black_box(i as f32) / 1000.0; 8];
            let v = [black_box(i as f32) / 2000.0; 8];
            cache.observe(black_box(&k), black_box(&v), 0.5, 0.5);
            i = i.wrapping_add(1);
        });
    });

    group.bench_function("heap_w16", |b| {
        let mut cache: HippocampalCache<8, 16> = HippocampalCache::new_with_ones_gamma();
        let mut i = 0u32;
        b.iter(|| {
            let k = [black_box(i as f32) / 1000.0; 8];
            let v = [black_box(i as f32) / 2000.0; 8];
            cache.observe(black_box(&k), black_box(&v), 0.5, 0.5);
            i = i.wrapping_add(1);
        });
    });

    group.finish();
}

criterion_group!(benches, bench_observe, bench_read, bench_heap_vs_sorted);
criterion_main!(benches);
