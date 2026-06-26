//! TriggeredInjectionGate benchmark (Plan 278 Phase 2, T2.8).
//!
//! Measures `EntropyThresholdGate::should_inject` latency.
//! Target: **<10ns p99** (one sigmoid + one compare).
//!
//! Follows the existing katgpt-rs bench convention (`std::time::Instant`,
//! `harness = false`, `fn main()`) — criterion is not a katgpt-rs dev-dep.
//!
//! Run with:
//! ```bash
//! cargo bench --bench triggered_injection_bench --features faithfulness_probe,triggered_injection
//! ```
//! Or as a plain binary:
//! ```bash
//! cargo run --release --bench triggered_injection_bench --features faithfulness_probe,triggered_injection
//! ```

use katgpt_core::faithfulness::gate::{EntropyThresholdGate, TriggeredInjectionGate};

fn main() {
    println!("=== TriggeredInjectionGate Benchmark (Plan 278 T2.8) ===\n");
    println!("Target: <10ns p99 for `should_inject` (sigmoid + compare).\n");

    let gate = EntropyThresholdGate::default();
    println!(
        "Config: tau={} lambda={} (default sigmoid slope)\n",
        gate.tau, gate.lambda
    );

    // 1. Functional sweep — show inject/skip decisions across the uncertainty range.
    println!("{:>12} {:>10} {:>14}", "uncertainty", "decision", "sigmoid_val");
    for &u in &[0.0f32, 0.1, 0.3, 0.49, 0.5, 0.51, 0.7, 0.9, 1.0] {
        let inject = gate.should_inject(u);
        // Reproduce the internal sigmoid for display (sigma(lambda * (u - tau))).
        let arg = gate.lambda * (u - gate.tau);
        let sig = if arg > 40.0 {
            1.0
        } else if arg < -40.0 {
            0.0
        } else {
            1.0 / (1.0 + (-arg).exp())
        };
        println!("{:>12.3} {:>10} {:>14.6}", u, inject as u8, sig);
    }
    println!();

    // 2. Latency measurement — 10M calls, sweep uncertainty across [0,1].
    let iters = 10_000_000;

    // Warm up the cache and branch predictor.
    let mut sink = false;
    for i in 0..1000 {
        let u = (i as f32) / 1000.0;
        sink ^= gate.should_inject(u);
    }

    let start = std::time::Instant::now();
    for i in 0..iters {
        // Sweep u across [0, 1] to exercise both inject and skip paths.
        let u = (i as f32) / (iters as f32);
        sink ^= gate.should_inject(u);
    }
    let elapsed = start.elapsed();
    let ns_per_call = elapsed.as_nanos() as f64 / iters as f64;

    println!(
        "should_inject: {} calls in {:?} → {:.3} ns/call (sink={})",
        iters, elapsed, ns_per_call, sink
    );

    // 3. Verdict.
    let target_ns = 10.0;
    let verdict = if ns_per_call < target_ns {
        "PASS ✅"
    } else {
        "FAIL ❌ (over 10ns budget)"
    };
    println!(
        "\nGOAT gate: {:.3} ns/call vs target <{}ns → {}",
        ns_per_call, target_ns, verdict
    );

    // 4. p99-style measurement: 100 batches of 100k, report the slowest batch.
    let batch_size = 100_000;
    let n_batches = 100;
    let mut batch_max_ns: u128 = 0;
    for b in 0..n_batches {
        let bstart = std::time::Instant::now();
        for i in 0..batch_size {
            let u = ((b * batch_size + i) as f32) / (n_batches * batch_size) as f32;
            sink ^= gate.should_inject(u);
        }
        let bns = bstart.elapsed().as_nanos();
        if bns > batch_max_ns {
            batch_max_ns = bns;
        }
    }
    let p99_ns = batch_max_ns as f64 / batch_size as f64;
    println!(
        "p99 (slowest of {} batches × {} calls): {:.3} ns/call (sink={})",
        n_batches, batch_size, p99_ns, sink
    );
}
