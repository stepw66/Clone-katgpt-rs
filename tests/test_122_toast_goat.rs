//! GOAT proofs for ToaST split tree tokenizer (Plan 122).
//!
//! **Source:** Schmidt et al. (2026). Tokenization with Split Trees. arXiv:2605.22705

#[cfg(feature = "toast_tokenizer")]
mod tests {
    use std::collections::HashMap;

    use katgpt_tokenizer::{
        BpeTokenizerImpl, BpeTrainer, SplitTreeBuilder, ToastTokenizer, ToastTokenizerImpl,
    };

    fn make_simple_tokenizer() -> ToastTokenizer {
        let mut vocab_to_id = HashMap::new();
        let mut id_to_vocab = Vec::new();

        // Special tokens
        for (id, tok) in [
            (0usize, b"<pad>".to_vec()),
            (1, b"<bos>".to_vec()),
            (2, b"<eos>".to_vec()),
            (3, b"<unk>".to_vec()),
        ] {
            vocab_to_id.insert(tok.clone(), id);
            id_to_vocab.push(tok);
        }

        // All ASCII bytes
        for b in 0u8..=127 {
            let bytes = vec![b];
            if !vocab_to_id.contains_key(&bytes) {
                let id = id_to_vocab.len();
                vocab_to_id.insert(bytes.clone(), id);
                id_to_vocab.push(bytes);
            }
        }

        // Common words
        for word in ["hello", "world", "the", "test", "split"] {
            let bytes = word.as_bytes().to_vec();
            if !vocab_to_id.contains_key(&bytes) {
                let id = id_to_vocab.len();
                vocab_to_id.insert(bytes.clone(), id);
                id_to_vocab.push(bytes);
            }
        }

        ToastTokenizer {
            vocab_to_id,
            id_to_vocab,
            trees: HashMap::new(),
            bos_id: 1,
            eos_id: 2,
            pad_id: 0,
            unk_id: 3,
            datrie_vocab: None,
        }
    }

    fn make_ngram_counts() -> HashMap<Vec<u8>, u64> {
        let mut counts = HashMap::new();
        for word in ["hello", "world", "the", "test", "split"] {
            let bytes = word.as_bytes();
            counts.insert(bytes.to_vec(), 1000);
            for len in 2..=bytes.len() {
                for start in 0..=bytes.len() - len {
                    let sub = bytes[start..start + len].to_vec();
                    *counts.entry(sub).or_insert(0) += 100;
                }
            }
        }
        counts
    }

    /// Build a ToaST tokenizer from a corpus (shared with test_120 pattern).
    fn build_toast_from_corpus(corpus: &str) -> ToastTokenizer {
        let bytes = corpus.as_bytes();

        let mut ngram_counts: HashMap<Vec<u8>, u64> = HashMap::new();
        for n in 2..=3 {
            for i in 0..bytes.len().saturating_sub(n - 1) {
                *ngram_counts.entry(bytes[i..i + n].to_vec()).or_default() += 1;
            }
        }

        let mut vocab_to_id = HashMap::new();
        let mut id_to_vocab = Vec::new();

        for (id, tok) in [
            (0usize, b"<pad>".to_vec()),
            (1, b"<bos>".to_vec()),
            (2, b"<eos>".to_vec()),
            (3, b"<unk>".to_vec()),
        ] {
            vocab_to_id.insert(tok.clone(), id);
            id_to_vocab.push(tok);
        }

        for b in 0u8..=127 {
            let bv = vec![b];
            if !vocab_to_id.contains_key(&bv) {
                let id = id_to_vocab.len();
                vocab_to_id.insert(bv.clone(), id);
                id_to_vocab.push(bv);
            }
        }

        let mut word_counts: HashMap<&str, usize> = HashMap::new();
        for word in corpus.split_whitespace() {
            *word_counts.entry(word).or_default() += 1;
        }

        let mut trees = HashMap::new();
        let builder = SplitTreeBuilder::new(&ngram_counts, 1);

        for (word, &count) in &word_counts {
            if count >= 2 && word.len() >= 2 {
                let wb = word.as_bytes();
                if !vocab_to_id.contains_key(wb) {
                    let id = id_to_vocab.len();
                    vocab_to_id.insert(wb.to_vec(), id);
                    id_to_vocab.push(wb.to_vec());
                }
                let tree = builder.build(wb);
                trees.insert(wb.to_vec(), tree);
            }
        }

        ToastTokenizer {
            vocab_to_id,
            id_to_vocab,
            trees,
            bos_id: 1,
            eos_id: 2,
            pad_id: 0,
            unk_id: 3,
            datrie_vocab: None,
        }
    }

    /// Count single-byte tokens in a token ID list (tokens whose byte length == 1).
    fn count_single_byte_tokens_toast(tokenizer: &ToastTokenizer, ids: &[usize]) -> usize {
        ids.iter()
            .filter(|&&id| tokenizer.id_to_vocab.get(id).is_some_and(|v| v.len() == 1))
            .count()
    }

    /// Count single-byte tokens in a BPE token ID list.
    fn count_single_byte_tokens_bpe(
        tokenizer: &katgpt_tokenizer::BpeTokenizer,
        ids: &[usize],
    ) -> usize {
        ids.iter()
            .filter(|&&id| tokenizer.id_to_vocab.get(id).is_some_and(|v| v.len() == 1))
            .count()
    }

    fn make_tokenizer_with_trees() -> ToastTokenizer {
        let mut tokenizer = make_simple_tokenizer();
        let counts = make_ngram_counts();
        let builder = SplitTreeBuilder::new(&counts, 10);

        for word in ["hello", "world", "the", "test", "split"] {
            let bytes = word.as_bytes();
            let tree = builder.build(bytes);
            tokenizer.trees.insert(bytes.to_vec(), tree);
        }

        tokenizer
    }

    // ── G1: Split tree builder basics ──────────────────────────

    #[test]
    fn test_split_tree_builder_basic() {
        let counts = make_ngram_counts();
        let builder = SplitTreeBuilder::new(&counts, 10);

        let tree = builder.build("hello".as_bytes());
        assert_eq!(tree.pretoken, b"hello");
        assert!(!tree.nodes.is_empty());
    }

    #[test]
    fn test_split_tree_single_byte() {
        let counts = make_ngram_counts();
        let builder = SplitTreeBuilder::new(&counts, 10);

        let tree = builder.build(b"a");
        assert_eq!(tree.pretoken, b"a");
        assert_eq!(tree.nodes.len(), 1);
        assert!(tree.nodes[0].left.is_none());
        assert!(tree.nodes[0].right.is_none());
    }

    #[test]
    fn test_split_tree_empty() {
        let counts = make_ngram_counts();
        let builder = SplitTreeBuilder::new(&counts, 10);

        let tree = builder.build(b"");
        assert!(tree.pretoken.is_empty());
        assert!(tree.nodes.is_empty());
    }

    #[test]
    fn test_split_tree_root_covers_pretoken() {
        let counts = make_ngram_counts();
        let builder = SplitTreeBuilder::new(&counts, 10);

        let tree = builder.build("test".as_bytes());
        assert!(!tree.nodes.is_empty());
        // Root node covers the entire pretoken
        assert_eq!(tree.nodes[0].start, 0);
        assert_eq!(tree.nodes[0].end, 4);
    }

    #[test]
    fn test_split_tree_ngram_guided_split() {
        let counts = make_ngram_counts();
        let builder = SplitTreeBuilder::new(&counts, 10);

        // "hello" has high n-gram counts — tree should prefer known substrings
        let tree = builder.build("hello".as_bytes());
        // Root must have children since length > 1
        assert!(tree.nodes[0].left.is_some());
        assert!(tree.nodes[0].right.is_some());
    }

    // ── G2: Encode/decode roundtrip ────────────────────────────

    #[test]
    fn test_encode_empty_string() {
        let tokenizer = make_simple_tokenizer();
        let ids = ToastTokenizerImpl::encode(&tokenizer, "");
        assert!(ids.is_empty());
    }

    #[test]
    fn test_encode_decode_roundtrip_ascii() {
        let tokenizer = make_tokenizer_with_trees();
        let text = "hello world";
        let ids = ToastTokenizerImpl::encode(&tokenizer, text);
        let decoded = ToastTokenizerImpl::decode(&tokenizer, &ids);
        assert_eq!(decoded, text, "Roundtrip must be identity on ASCII");
    }

    #[test]
    fn test_encode_decode_roundtrip_multi_word() {
        let tokenizer = make_tokenizer_with_trees();
        let text = "the test split hello world";
        let ids = ToastTokenizerImpl::encode(&tokenizer, text);
        let decoded = ToastTokenizerImpl::decode(&tokenizer, &ids);
        assert_eq!(
            decoded, text,
            "Roundtrip must be identity on multi-word ASCII"
        );
    }

    #[test]
    fn test_encode_decode_roundtrip_single_char() {
        let tokenizer = make_tokenizer_with_trees();
        let text = "a";
        let ids = ToastTokenizerImpl::encode(&tokenizer, text);
        let decoded = ToastTokenizerImpl::decode(&tokenizer, &ids);
        assert_eq!(decoded, text, "Roundtrip must be identity on single char");
    }

    // ── G3: Known words encode as single tokens ───────────────

    #[test]
    fn test_encode_known_word() {
        let tokenizer = make_tokenizer_with_trees();
        let ids = ToastTokenizerImpl::encode(&tokenizer, "hello");
        assert_eq!(ids.len(), 1, "Known word should encode as single token");
    }

    #[test]
    fn test_encode_all_known_words_single_token() {
        let tokenizer = make_tokenizer_with_trees();
        for word in ["hello", "world", "the", "test", "split"] {
            let ids = ToastTokenizerImpl::encode(&tokenizer, word);
            assert_eq!(ids.len(), 1, "Word '{word}' should encode as single token");
        }
    }

    // ── G4: No unknown tokens for ASCII ────────────────────────

    #[test]
    fn test_no_unknown_tokens_for_ascii() {
        let tokenizer = make_tokenizer_with_trees();
        let text = "the test split hello world";
        let ids = ToastTokenizerImpl::encode(&tokenizer, text);
        let unk_count = ids.iter().filter(|&&id| id == tokenizer.unk_id).count();
        assert_eq!(unk_count, 0, "All ASCII bytes should be in vocab");
    }

    #[test]
    fn test_no_unknown_tokens_for_ascii_bytes() {
        let tokenizer = make_simple_tokenizer();
        // Test all printable ASCII
        let text: String = (32u8..=126).map(char::from).collect();
        let ids = ToastTokenizerImpl::encode(&tokenizer, &text);
        let unk_count = ids.iter().filter(|&&id| id == tokenizer.unk_id).count();
        assert_eq!(unk_count, 0, "All printable ASCII should be in vocab");
    }

    // ── G5: Decode unknown IDs produce replacement ─────────────

    #[test]
    fn test_decode_out_of_range_id() {
        let tokenizer = make_simple_tokenizer();
        let decoded = ToastTokenizerImpl::decode(&tokenizer, &[9999]);
        assert_eq!(decoded, "", "Out-of-range ID should produce empty string");
    }

    // ── G6: Compression vs byte-level ──────────────────────────

    #[test]
    fn test_compression_fewer_tokens_than_bytes() {
        let tokenizer = make_tokenizer_with_trees();
        let text = "hello world the test split";
        let ids = ToastTokenizerImpl::encode(&tokenizer, text);
        // Each known word = 1 token, spaces = 1 token each
        // 5 words + 4 spaces = 9 tokens, vs 26 bytes
        assert!(
            ids.len() < text.len(),
            "Tokenized output ({ids_len} tokens) should be shorter than raw bytes ({text_len})",
            ids_len = ids.len(),
            text_len = text.len(),
        );
    }

    // ── G7: Serde roundtrip ────────────────────────────────────

    #[test]
    fn test_serde_roundtrip() {
        let tokenizer = make_tokenizer_with_trees();
        let json = serde_json::to_string(&tokenizer).expect("serialize");
        let restored: ToastTokenizer = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(restored.vocab_size(), tokenizer.vocab_size());
        assert_eq!(restored.bos_id, tokenizer.bos_id);
        assert_eq!(restored.eos_id, tokenizer.eos_id);
        assert_eq!(restored.pad_id, tokenizer.pad_id);
        assert_eq!(restored.unk_id, tokenizer.unk_id);
        assert_eq!(restored.trees.len(), tokenizer.trees.len());
    }

    #[test]
    fn test_serde_roundtrip_encode_identity() {
        let tokenizer = make_tokenizer_with_trees();
        let json = serde_json::to_string(&tokenizer).expect("serialize");
        let restored: ToastTokenizer = serde_json::from_str(&json).expect("deserialize");

        let text = "hello world";
        let ids_orig = ToastTokenizerImpl::encode(&tokenizer, text);
        let ids_restored = ToastTokenizerImpl::encode(&restored, text);
        assert_eq!(
            ids_orig, ids_restored,
            "Serde roundtrip must preserve encoding"
        );
    }

    // ── G2: Single-byte fallback comparison (T5 G2) ───────────

    #[test]
    fn proof_t5_g2_toast_single_byte_fallback_leq_bpe() {
        // Corpus has every test word repeated >= 2× so ToaST builds trees for all of them.
        // This mirrors real usage where the tokenizer is trained on the target domain.
        let corpus = "the cat sat on the mat the cat the mat the test hello world the test split \
                      cat cat sat sat mat mat on on \
                      hello world test split hello world test split \
                      the cat the mat the test hello world the cat the mat the test split \
                      the cat the mat the test hello world the cat the mat the test split \
                      cat cat sat mat on hello world test split";
        let bpe = BpeTrainer::train(corpus, 300);
        let toast = build_toast_from_corpus(corpus);

        let test_strings = [
            "the cat sat on the mat",
            "hello world test split",
            "the test hello world",
            "cat mat on sat",
        ];

        println!("┌──────────────────────────────────────┬────────────────┬─────────────────┐");
        println!("│ text                                 │ bpe_single_byte│ toast_single_by │");
        println!("├──────────────────────────────────────┼────────────────┼─────────────────┤");

        for text in &test_strings {
            let bpe_ids = BpeTokenizerImpl::encode(&bpe, text);
            let toast_ids = ToastTokenizerImpl::encode(&toast, text);

            let bpe_single = count_single_byte_tokens_bpe(&bpe, &bpe_ids);
            let toast_single = count_single_byte_tokens_toast(&toast, &toast_ids);

            println!("│ {text:<36} │ {bpe_single:>14} │ {toast_single:>15} │");

            assert!(
                toast_single <= bpe_single,
                "ToaST single-byte fallback ({toast_single}) must be <= BPE ({bpe_single}) for \"{text}\""
            );
        }

        println!("└──────────────────────────────────────┴────────────────┴─────────────────┘");
    }

    // ── G3: Inference latency benchmark (T5 G3) ──────────────

    #[test]
    fn proof_t5_g3_toast_latency_leq_2x_bpe() {
        let corpus = "the cat sat on the mat the cat the mat the test hello world the test split \
                      the cat the mat the test hello world the cat the mat the test split \
                      hello world test split hello world test split \
                      the cat the mat the test hello world the cat the mat the test split \
                      the cat sat on the mat the cat the mat the test";
        let bpe = BpeTrainer::train(corpus, 300);
        let toast = build_toast_from_corpus(corpus);

        let test_strings = [
            "the cat sat on the mat",
            "hello world test split",
            "the cat the mat the test hello world the cat the mat the test",
        ];

        // Warmup
        for text in &test_strings {
            let _ = BpeTokenizerImpl::encode(&bpe, text);
            let _ = ToastTokenizerImpl::encode(&toast, text);
        }

        println!("┌──────────────────────────────────────┬────────────┬──────────────┬──────────┐");
        println!("│ text                                 │ bpe_us     │ toast_us     │ ratio    │");
        println!("├──────────────────────────────────────┼────────────┼──────────────┼──────────┤");

        for text in &test_strings {
            // BPE timing — multiple iterations for stable measurement
            let iters = 1000u32;
            let bpe_start = std::time::Instant::now();
            for _ in 0..iters {
                let ids = BpeTokenizerImpl::encode(&bpe, text);
                std::hint::black_box(&ids);
            }
            let bpe_elapsed = bpe_start.elapsed();

            // ToaST timing
            let toast_start = std::time::Instant::now();
            for _ in 0..iters {
                let ids = ToastTokenizerImpl::encode(&toast, text);
                std::hint::black_box(&ids);
            }
            let toast_elapsed = toast_start.elapsed();

            let bpe_us = bpe_elapsed.as_secs_f64() * 1e6;
            let toast_us = toast_elapsed.as_secs_f64() * 1e6;
            let ratio = toast_us / bpe_us;

            println!("│ {text:<36} │ {bpe_us:>10.1} │ {toast_us:>12.1} │ {ratio:>8.3} │");

            assert!(
                ratio <= 2.0,
                "ToaST latency ({toast_us:.1}us) must be <= 2× BPE ({bpe_us:.1}us), ratio={ratio:.3}"
            );
        }

        println!("└──────────────────────────────────────┴────────────┴──────────────┴──────────┘");
    }
}
