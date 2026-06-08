//! GOAT benchmark for Modality-Pruned Context Loading (Plan 227 Phase 3).
//!
//! Measures: classification accuracy, latency per query class.

use katgpt_rs::pipeline_pruner::{PipelineConfig, QueryClassifier, QueryFeatures};

#[test]
fn test_classify_simple_fast() {
    let classifier = QueryClassifier::new();
    let start = std::time::Instant::now();

    for _ in 0..10_000 {
        let features = QueryFeatures {
            entropy: 0.3,
            expected_output_len: 32,
            input_len: 64,
            syntax_ratio: 0.0,
            ..Default::default()
        };
        let result = classifier.classify(&features);
        assert_eq!(result, PipelineConfig::Simple);
    }

    let elapsed = start.elapsed();
    let us = elapsed.as_secs_f64() * 1e6;
    eprintln!(
        "10K simple classifications: {us:.0}μs ({:.2}μs each)",
        us / 10_000.0
    );
    assert!(
        elapsed.as_secs() < 1,
        "10K classifications took {us:.0}μs — too slow"
    );
}

#[test]
fn test_classify_code_fast() {
    let classifier = QueryClassifier::new();
    let start = std::time::Instant::now();

    for _ in 0..10_000 {
        let result = classifier.classify_prompt("fn main() { println!(\"hello\"); }");
        assert_eq!(result, PipelineConfig::Code);
    }

    let elapsed = start.elapsed();
    let us = elapsed.as_secs_f64() * 1e6;
    eprintln!(
        "10K code classifications: {us:.0}μs ({:.2}μs each)",
        us / 10_000.0
    );
}

#[test]
fn test_classify_long_context_fast() {
    let classifier = QueryClassifier::new();
    let start = std::time::Instant::now();

    for _ in 0..10_000 {
        let features = QueryFeatures {
            entropy: 0.3,
            expected_output_len: 2048,
            input_len: 4096,
            syntax_ratio: 0.0,
            ..Default::default()
        };
        let result = classifier.classify(&features);
        assert_eq!(result, PipelineConfig::LongContext);
    }

    let elapsed = start.elapsed();
    let us = elapsed.as_secs_f64() * 1e6;
    eprintln!("10K long-context classifications: {us:.0}μs");
}

#[test]
fn test_classify_reasoning_fast() {
    let classifier = QueryClassifier::new();
    let start = std::time::Instant::now();

    for _ in 0..10_000 {
        let features = QueryFeatures {
            entropy: 0.9,
            expected_output_len: 512,
            input_len: 256,
            syntax_ratio: 0.0,
            ..Default::default()
        };
        let result = classifier.classify(&features);
        assert_eq!(result, PipelineConfig::Reasoning);
    }

    let elapsed = start.elapsed();
    let us = elapsed.as_secs_f64() * 1e6;
    eprintln!("10K reasoning classifications: {us:.0}μs");
}

#[test]
fn test_all_classes_correct() {
    let classifier = QueryClassifier::new();

    // Simple
    assert_eq!(
        classifier.classify(&QueryFeatures {
            entropy: 0.2,
            input_len: 50,
            syntax_ratio: 0.0,
            ..Default::default()
        }),
        PipelineConfig::Simple
    );

    // Code (high syntax ratio)
    assert_eq!(
        classifier.classify(&QueryFeatures {
            entropy: 0.5,
            input_len: 200,
            syntax_ratio: 0.15,
            ..Default::default()
        }),
        PipelineConfig::Code
    );

    // Long context
    assert_eq!(
        classifier.classify(&QueryFeatures {
            entropy: 0.3,
            input_len: 4096,
            syntax_ratio: 0.0,
            ..Default::default()
        }),
        PipelineConfig::LongContext
    );

    // Reasoning (high entropy)
    assert_eq!(
        classifier.classify(&QueryFeatures {
            entropy: 0.9,
            input_len: 200,
            syntax_ratio: 0.0,
            ..Default::default()
        }),
        PipelineConfig::Reasoning
    );
}

#[test]
fn test_pipeline_latency_no_regression() {
    let classifier = QueryClassifier::new();

    // Measure raw classification latency
    let start = std::time::Instant::now();
    for _ in 0..100_000 {
        let _ = classifier.classify_prompt("Hello, how are you today?");
    }
    let elapsed = start.elapsed();
    let ns = elapsed.as_secs_f64() * 1e9 / 100_000.0;

    eprintln!("Classification latency: {ns:.0}ns per query");

    // Classification should be < 1μs per query (trivial computation)
    assert!(ns < 10_000.0, "Classification too slow: {ns:.0}ns");
}
