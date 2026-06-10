//! Pipeline Pruner — modality-aware inference pipeline selection.
//! Classifies queries and selects the optimal inference pipeline.
//! Feature-gated behind `modality_pruned_load`.

/// Inference pipeline configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum PipelineConfig {
    /// Direct decode only — no DDTree, no speculative, no KV compression.
    Simple = 0,
    /// DDTree + SynPruner, no KV compression — for code generation.
    Code = 1,
    /// VortexFlow + KV compression, no speculative — for long context.
    LongContext = 2,
    /// Adaptive CoT + ThoughtFold, full precision — for reasoning.
    Reasoning = 3,
}

/// Query features used for classification.
#[derive(Debug, Clone)]
pub struct QueryFeatures {
    /// Shannon entropy of the input distribution.
    pub entropy: f32,
    /// Expected output length (from heuristics).
    pub expected_output_len: usize,
    /// Input prompt length in tokens.
    pub input_len: usize,
    /// Ratio of syntactic tokens (brackets, semicolons, etc.) to total.
    pub syntax_ratio: f32,
    /// River Valley signal strength (if available).
    pub rv_signal: Option<f32>,
}

impl Default for QueryFeatures {
    fn default() -> Self {
        Self {
            entropy: 0.5,
            expected_output_len: 64,
            input_len: 32,
            syntax_ratio: 0.0,
            rv_signal: None,
        }
    }
}

/// Classifies queries into pipeline configurations.
#[derive(Debug, Clone)]
pub struct QueryClassifier {
    /// Entropy threshold for reasoning classification.
    pub entropy_threshold: f32,
    /// Input length threshold for long context classification.
    pub long_context_threshold: usize,
    /// Syntax ratio threshold for code classification.
    pub code_syntax_threshold: f32,
}

impl Default for QueryClassifier {
    fn default() -> Self {
        Self {
            entropy_threshold: 0.7,
            long_context_threshold: 2048,
            code_syntax_threshold: 0.05,
        }
    }
}

impl QueryClassifier {
    pub fn new() -> Self {
        Self::default()
    }

    /// Classify a query into a pipeline configuration.
    /// Uses sigmoid-gated scoring for smooth transitions.
    pub fn classify(&self, features: &QueryFeatures) -> PipelineConfig {
        let code_score = sigmoid((features.syntax_ratio - self.code_syntax_threshold) * 20.0);
        let long_score =
            sigmoid((features.input_len as f32 - self.long_context_threshold as f32) / 500.0);
        let reasoning_score = sigmoid((features.entropy - self.entropy_threshold) * 10.0);

        // Priority: Code > LongContext > Reasoning > Simple
        if code_score > 0.6 {
            PipelineConfig::Code
        } else if long_score > 0.6 {
            PipelineConfig::LongContext
        } else if reasoning_score > 0.6 {
            PipelineConfig::Reasoning
        } else {
            PipelineConfig::Simple
        }
    }

    /// Classify from raw prompt text (basic heuristic).
    pub fn classify_prompt(&self, prompt: &str) -> PipelineConfig {
        let syntax_count = prompt
            .chars()
            .filter(|c| matches!(c, '(' | ')' | '{' | '}' | '[' | ']' | ';' | '=' | '<' | '>'))
            .count();

        let features = QueryFeatures {
            entropy: 0.5, // default
            expected_output_len: prompt.len(),
            input_len: prompt.len(),
            syntax_ratio: syntax_count as f32 / prompt.len().max(1) as f32,
            rv_signal: None,
        };

        self.classify(&features)
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classify_simple() {
        let classifier = QueryClassifier::new();
        let features = QueryFeatures {
            entropy: 0.3,
            expected_output_len: 32,
            input_len: 64,
            syntax_ratio: 0.0,
            ..Default::default()
        };
        assert_eq!(classifier.classify(&features), PipelineConfig::Simple);
    }

    #[test]
    fn test_classify_code() {
        let classifier = QueryClassifier::new();
        let features = QueryFeatures {
            entropy: 0.5,
            expected_output_len: 128,
            input_len: 256,
            syntax_ratio: 0.15, // high syntax ratio
            ..Default::default()
        };
        assert_eq!(classifier.classify(&features), PipelineConfig::Code);
    }

    #[test]
    fn test_classify_long_context() {
        let classifier = QueryClassifier::new();
        let features = QueryFeatures {
            entropy: 0.3,
            expected_output_len: 2048,
            input_len: 4096,
            syntax_ratio: 0.0,
            ..Default::default()
        };
        assert_eq!(classifier.classify(&features), PipelineConfig::LongContext);
    }

    #[test]
    fn test_classify_reasoning() {
        let classifier = QueryClassifier::new();
        let features = QueryFeatures {
            entropy: 0.9,
            expected_output_len: 512,
            input_len: 256,
            syntax_ratio: 0.0,
            ..Default::default()
        };
        assert_eq!(classifier.classify(&features), PipelineConfig::Reasoning);
    }

    #[test]
    fn test_classify_prompt_code() {
        let classifier = QueryClassifier::new();
        let result = classifier.classify_prompt("fn main() { println!(\"hello\"); }");
        assert_eq!(result, PipelineConfig::Code);
    }

    #[test]
    fn test_classify_prompt_simple() {
        let classifier = QueryClassifier::new();
        let result = classifier.classify_prompt("What is the weather today?");
        assert_eq!(result, PipelineConfig::Simple);
    }

    #[test]
    fn test_pipeline_config_repr() {
        assert_eq!(PipelineConfig::Simple as u8, 0);
        assert_eq!(PipelineConfig::Code as u8, 1);
        assert_eq!(PipelineConfig::LongContext as u8, 2);
        assert_eq!(PipelineConfig::Reasoning as u8, 3);
    }
}
