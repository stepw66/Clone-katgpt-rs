//! Plan 223 Example: LLMExecGuard Demo

fn main() {
    #[cfg(feature = "llmexec_guard")]
    {
        use katgpt_rs::llmexec_guard::{LlmExecGuardConfig, llmexec_confidence, verify_tier};

        println!("=== Plan 223: LLMExecGuard Demo ===\n");
        let config = LlmExecGuardConfig::default();

        // Simulate different entropy/depth scenarios
        let scenarios = [
            ("Low entropy, shallow", 0.1, 1),
            ("Medium entropy", 0.5, 3),
            ("High entropy, deep", 0.9, 7),
        ];

        for (name, entropy, depth) in scenarios {
            let conf = llmexec_confidence(entropy, depth, &config);
            let tier = verify_tier(entropy, depth, &config);
            println!(
                "  {}: entropy={}, depth={}, conf={:.4}, tier={:?}",
                name, entropy, depth, conf, tier
            );
        }

        println!("\nDone.");
    }

    #[cfg(not(feature = "llmexec_guard"))]
    println!("Enable feature: cargo run --example llmexec_guard_demo --features llmexec_guard");
}
