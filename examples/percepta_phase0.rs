//! Phase 0 feasibility measurement for "transformer-VM in the browser".
//!
//! Proves the analytically-built transformer actually executes a compiled
//! program natively, and measures the three numbers that decide whether a
//! browser (wasm) run is viable:
//!   1. weight artifact size (bytes) + model dims,
//!   2. program token count (prefix),
//!   3. inference time + throughput (tok/s) for a tiny program.
//!
//! Run: cargo run --release --features percepta_compile --example percepta_phase0

use std::time::Instant;

use katgpt_rs::percepta::compile::{compile_rust_program, rust_template};
use katgpt_rs::percepta::runner::Runner;
use katgpt_rs::percepta::weights::TransformerWeights;

fn mat_f64(m: &[Vec<f64>]) -> usize {
    m.iter().map(|r| r.len()).sum()
}

fn weight_f64_count(w: &TransformerWeights) -> usize {
    let mut n = mat_f64(&w.embedding) + mat_f64(&w.unembedding);
    for l in &w.layers {
        n += mat_f64(&l.attention.in_proj) + mat_f64(&l.attention.out_proj);
        n += mat_f64(&l.ffn.ff_in) + mat_f64(&l.ffn.ff_out);
    }
    n
}

fn parse_prefix_tokens(prefix: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    for line in prefix.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if line == "{" || line == "}" {
            tokens.push(line.to_string());
            continue;
        }
        for part in line.split_whitespace() {
            tokens.push(part.to_string());
        }
    }
    tokens
}

fn extract_output(tokens: &[String]) -> String {
    tokens
        .iter()
        .filter_map(|t| {
            if t.starts_with("out(") && t.ends_with(')') {
                let inner = &t[4..t.len() - 1];
                if inner.len() == 1 {
                    inner.chars().next()
                } else {
                    u8::from_str_radix(inner, 16).ok().map(|b| b as char)
                }
            } else {
                None
            }
        })
        .collect()
}

fn bar() {
    println!("{}", "─".repeat(64));
}

fn main() {
    println!("🔬 Percepta Phase 0 — transformer-VM feasibility");
    bar();

    // ── 1. Build the universal interpreter transformer (native, MILP) ──
    let t0 = Instant::now();
    let build = match Runner::build(None) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("❌ build failed: {e}");
            return;
        }
    };
    let build_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let f64s = weight_f64_count(&build.weights);
    let bytes_f64 = f64s * 8;
    let bytes_f32 = f64s * 4;
    println!("BUILD (done once, native — clang/MILP not needed in browser)");
    println!("  d_model   : {}", build.config.d_model);
    println!("  n_layers  : {}", build.config.n_layers);
    println!("  vocab     : {}", build.vocab.len());
    println!("  weights   : {f64s} f64  =  {:.2} MB (f64) / {:.2} MB (f32)",
        bytes_f64 as f64 / 1e6, bytes_f32 as f64 / 1e6);
    println!("  build time: {build_ms:.0} ms");
    bar();

    // ── 2. Compile a tiny program → token prefix ──
    // Trivial: output two bytes. Smallest real program to prove execution.
    let src = rust_template("output_byte(b'H' as i32); output_byte(b'i' as i32);");
    let compiled = match compile_rust_program(&src, "") {
        Ok(c) => c,
        Err(e) => {
            eprintln!("❌ compile failed (needs rustc + wasm32): {e}");
            return;
        }
    };
    let prefix_tokens = parse_prefix_tokens(&compiled.prefix);
    println!("COMPILE (tiny program: print \"Hi\")");
    println!("  instructions : {}", compiled.program.len());
    println!("  prefix tokens: {}", prefix_tokens.len());
    bar();

    // ── 3. Run it through the transformer (this is the real proof) ──
    let t1 = Instant::now();
    let gr = match Runner::run(&build, &prefix_tokens, 50_000) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("❌ run failed: {e}");
            return;
        }
    };
    let run_s = t1.elapsed().as_secs_f64();
    let n = gr.tokens.len();
    let tps = n as f64 / run_s;
    let output = extract_output(&gr.tokens);
    println!("RUN (transformer forward pass per token — native)");
    println!("  tokens generated: {n}");
    println!("  time            : {:.1} ms", run_s * 1000.0);
    println!("  throughput      : {tps:.0} tok/s (native)");
    println!("  output          : {output:?}");
    bar();

    // ── 4. Extrapolate to Sudoku (~900K tokens per Percepta) ──
    let sudoku_tokens = 900_000.0;
    println!("EXTRAPOLATION → Sudoku (~900K tokens)");
    println!("  native est.     : {:.1} s", sudoku_tokens / tps);
    println!("  browser est.    : {:.1}–{:.1} s  (assume 3–10× slower in wasm)",
        sudoku_tokens / tps * 3.0, sudoku_tokens / tps * 10.0);
    println!("  weights ship    : {:.2} MB (f32)", bytes_f32 as f64 / 1e6);
    bar();
    println!(
        "✅ verdict: {}",
        if output.contains("Hi") {
            "transformer-VM executes correctly. Numbers above decide Sudoku viability."
        } else {
            "ran, but output mismatch — inspect tokens before proceeding."
        }
    );
}
