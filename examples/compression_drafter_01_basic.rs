//! CompressionDrafter basic example — corpus-as-model quest generation (Plan 285).
//!
//! Run with: cargo run --example compression_drafter_01_basic --features compression_drafter

use katgpt_core::compression_drafter::{CompressionDrafter, Lz4FlexDrafter};

fn main() {
    println!("=== CompressionDrafter — corpus-as-model (Plan 285) ===\n");

    // 1. Build corpus from 8 hardcoded S-V-O quest triples (same as TernaryDraftModel).
    //    Repeated 4× to give LZ4's hash table enough density to find matches
    //    (LZ4 minimum match = 4 bytes; needs corpus repetition to engage).
    let quest_templates = [
        "guard needs sword",
        "merchant wants potion",
        "king finds amulet",
        "sage seeks quest",
        "guard wants amulet",
        "merchant finds sword",
        "king needs quest",
        "sage seeks potion",
    ];
    let corpus: Vec<u8> = (0..4)
        .flat_map(|_| quest_templates.iter().map(|q| format!("{}\n", q)))
        .collect::<String>()
        .into_bytes();
    let mut drafter = Lz4FlexDrafter::new(corpus);

    println!(
        "Corpus: {} bytes, {} quest templates (×4 reps)",
        drafter.corpus().len(),
        quest_templates.len()
    );

    // 2. Score 4 candidate continuations given an empty context.
    //    All candidates are full quest lines of equal length — gzip-lm assumes
    //    fixed-length candidates (single-byte beam expansion in the original paper).
    //    Mixing lengths biases the score toward shorter candidates regardless of
    //    corpus match. We use 18-byte quest lines that appear verbatim in the corpus.
    let ctx: &[u8] = b"";
    let candidates: &[&[u8]] = &[
        b"guard needs sword\n",   // 18 bytes — appears 4× in corpus
        b"sage seeks potion\n",   // 18 bytes — appears 4× in corpus
        b"the dragon wakes\n\0\0",// 18 bytes — NOT in corpus (padding for length parity)
        b"zzzzzzzzzzzzzzzzzz\n",  // 18 bytes — all-unseen alphabet
    ];
    let scores = drafter.score_batch(ctx, candidates);

    println!("\nCandidate scores (higher = more compressible = more likely):");
    let worst = *scores.iter().min().unwrap();
    let best = *scores.iter().max().unwrap();
    let span = (best - worst).max(1);
    for (cand, score) in candidates.iter().zip(scores.iter()) {
        // Label relative to the score range — top-half = "seen-like", bottom-half = "unseen-like".
        let label = if (*score - worst) * 2 >= span { "seen-like (compresses well)" } else { "unseen-like (adds entropy)" };
        // Strip trailing '\n' / '\0' for display cleanliness.
        let display = String::from_utf8_lossy(cand)
            .trim_end_matches(|c: char| c == '\n' || c == '\0')
            .to_string();
        println!("  {:>20}  →  {:>4}  ({})", display, score, label);
    }

    // 3. Argmax pick — the candidate the compressor "remembers" best.
    let (best_idx, best_score) = scores.iter().enumerate().max_by_key(|&(_, &s)| s).unwrap();
    let best_display = String::from_utf8_lossy(candidates[best_idx])
        .trim_end_matches(|c: char| c == '\n' || c == '\0')
        .to_string();
    println!("\nArgmax: '{}' (score {})", best_display, best_score);

    // 4. Online learning: append the chosen candidate to the corpus.
    drafter.append(candidates[best_idx]);
    println!(
        "\nAfter append: corpus is now {} bytes",
        drafter.corpus().len()
    );

    // 5. The corpus IS the wired format — snapshot via BLAKE3 (or just bytes).
    let snapshot: &[u8] = drafter.corpus_bytes();
    println!(
        "Snapshot: {} bytes, ready for freeze/thaw commitment",
        snapshot.len()
    );

    println!("\nNo neural weights. No training. The compressor IS the model.");
}
