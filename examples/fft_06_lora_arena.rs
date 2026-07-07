//! Plan 296 Phase 7 T7.3 — SON-LT FFT Arena Evaluation.
//!
//! Loads a SON-LT (Sparse OPD-Native LoRA Training) trained LoRA adapter
//! and evaluates it in an N-round FFT tournament against the HLFFTPlayer
//! baseline.
//!
//! # Setup
//!
//! - Party 🧠 SON-LT  — LoRA-trained FFT Transformer (Plan 296 adapter)
//! - Enemy 🐵 HL      — bandit-adaptive baseline
//!
//! # GOAT Gate (Plan 296 T7.3)
//!
//! Target: SON-LT (Party) win rate ≥ 99% vs HL (Enemy).
//! The 30% training-time reduction is provided by SON-LT mask sparsity
//! (paper §4.4: ~5.7× speedup at 17.5% density).
//!
//! # Run
//!
//! ```sh
//! cargo run --example fft_06_lora_arena --features fft --release -- \
//!     --lora-path /path/to/fft_lora_sonlt_t73.bin --games 1000
//! ```

#![cfg(feature = "fft")]

use std::path::PathBuf;

use fastrand::Rng;

use katgpt_rs::pruners::fft::arena_runner::{FftArenaConfig, run_fft_battle};
use katgpt_rs::pruners::fft::players::HLFFTPlayer;
use katgpt_rs::pruners::fft::{FftLoRAPlayer, FftPlayer, TURN_LIMIT, Team};

// ── Constants ────────────────────────────────────────────────────

/// Default battle count for the tournament.
const DEFAULT_GAMES: usize = 1000;

/// SON-LT GOAT gate: Party win rate target (Plan 296 T7.3).
const SONLT_TARGET_WIN_RATE: f64 = 0.99;

/// Default LoRA path relative to CARGO_MANIFEST_DIR.
const DEFAULT_LORA_REL: &str = "../../../output/fft_lora_sonlt_t73.bin";

// ── CLI ──────────────────────────────────────────────────────────

struct Cli {
    lora_path: PathBuf,
    games: usize,
}

fn parse_args() -> Cli {
    let args: Vec<String> = std::env::args().collect();
    let default_lora = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(DEFAULT_LORA_REL);
    let mut cli = Cli {
        lora_path: default_lora,
        games: DEFAULT_GAMES,
    };

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--lora-path" if i + 1 < args.len() => {
                i += 1;
                cli.lora_path = PathBuf::from(&args[i]);
            }
            "--games" if i + 1 < args.len() => {
                i += 1;
                cli.games = args[i].parse().unwrap_or_else(|e| {
                    eprintln!("Bad --games value: {e}");
                    std::process::exit(1);
                });
            }
            "--help" | "-h" => {
                eprintln!("FFT SON-LT Arena (Plan 296 T7.3)");
                eprintln!();
                eprintln!("Options:");
                eprintln!(
                    "  --lora-path PATH  LoRA adapter file (default: {})",
                    DEFAULT_LORA_REL
                );
                eprintln!("  --games N         Battles to run (default: {DEFAULT_GAMES})");
                std::process::exit(0);
            }
            _ => {}
        }
        i += 1;
    }

    cli
}

// ── Main ─────────────────────────────────────────────────────────

fn main() {
    let cli = parse_args();

    println!("═╡ Plan 296 T7.3: FFT SON-LT Arena ╞═");
    println!();
    println!("  LoRA:    {}", cli.lora_path.display());
    println!("  Games:   {}", cli.games);
    println!(
        "  Target:  Party win rate ≥ {:.0}%",
        SONLT_TARGET_WIN_RATE * 100.0
    );
    println!();

    // Build party (SON-LT) — all 4 units share the same LoRA adapter.
    let lora_path = cli.lora_path.clone();
    let mut party: Vec<Box<dyn FftPlayer>> = vec![
        Box::new(FftLoRAPlayer::new_with_lora(0, &lora_path)),
        Box::new(FftLoRAPlayer::new_with_lora(1, &lora_path)),
        Box::new(FftLoRAPlayer::new_with_lora(2, &lora_path)),
        Box::new(FftLoRAPlayer::new_with_lora(3, &lora_path)),
    ];

    // Build enemy (HL baseline) — fresh Q-learners each battle.
    let make_enemy = || -> Vec<Box<dyn FftPlayer>> {
        vec![
            Box::new(HLFFTPlayer::new()),
            Box::new(HLFFTPlayer::new()),
            Box::new(HLFFTPlayer::new()),
            Box::new(HLFFTPlayer::new()),
        ]
    };

    // Report LoRA load status from player 0.
    let lora_active = party[0]
        .as_any_mut()
        .downcast_ref::<FftLoRAPlayer>()
        .is_some_and(|p| p.lora_active());
    println!(
        "  LoRA active: {}",
        if lora_active {
            "yes ✓"
        } else {
            "NO — running in heuristic-fallback mode"
        }
    );
    println!();

    if !lora_active {
        eprintln!("⚠ LoRA adapter did not load. Arena will measure heuristic baseline.");
        eprintln!("  This is still useful for sanity-checking the player wiring.");
        eprintln!();
    }

    // Run tournament.
    let config = FftArenaConfig {
        games: cli.games,
        turn_limit: TURN_LIMIT,
    };

    println!("Phase 1: Run {} battles", cli.games);

    let mut rng = Rng::with_seed(42);
    let start = std::time::Instant::now();

    let mut party_wins = 0usize;
    let mut enemy_wins = 0usize;
    let mut draws = 0usize;
    let mut total_ticks = 0u32;

    for game in 1..=cli.games {
        // Reset party players' state between games.
        for p in party.iter_mut() {
            p.reset();
        }
        // Fresh enemy party (HL state doesn't carry across matchups in our eval).
        let mut enemy = make_enemy();
        for p in enemy.iter_mut() {
            p.reset();
        }

        let result = run_fft_battle(&mut party, &mut enemy, config.turn_limit, &mut rng);
        total_ticks += result.ticks;

        match result.winner {
            Some(Team::Party) => party_wins += 1,
            Some(Team::Enemy) => enemy_wins += 1,
            None => draws += 1,
        }

        if game % 50 == 0 || game == cli.games {
            let elapsed = start.elapsed().as_secs_f32();
            let gps = game as f32 / elapsed.max(0.001);
            let party_wr = party_wins as f64 / game as f64 * 100.0;
            println!(
                "  [{game:>4}/{total}] P:{party_wins} E:{enemy_wins} D:{draws} | \
                 party WR: {party_wr:.1}% | avg ticks: {avg:.0} | {gps:.1} games/s",
                total = cli.games,
                avg = total_ticks as f64 / game as f64,
            );
        }
    }

    // Final report.
    let elapsed = start.elapsed();
    let party_wr = party_wins as f64 / cli.games as f64;
    let enemy_wr = enemy_wins as f64 / cli.games as f64;
    let draw_rate = draws as f64 / cli.games as f64;

    println!();
    println!("═╡ Results ╞═");
    println!();
    println!("  Total battles:   {}", cli.games);
    println!("  Party wins:      {party_wins} ({:.1}%)", party_wr * 100.0);
    println!("  Enemy wins:      {enemy_wins} ({:.1}%)", enemy_wr * 100.0);
    println!("  Draws:           {draws} ({:.1}%)", draw_rate * 100.0);
    println!(
        "  Avg ticks/battle: {:.1}",
        total_ticks as f64 / cli.games as f64
    );
    println!(
        "  Wall-clock:      {:.2}s ({:.1} games/s)",
        elapsed.as_secs_f64(),
        cli.games as f64 / elapsed.as_secs_f64().max(0.001)
    );
    println!();

    // GOAT gate verdict.
    println!("═╡ GOAT Gate Verdict (T7.3) ╞═");
    println!();
    if party_wr >= SONLT_TARGET_WIN_RATE {
        println!(
            "  ✓ PASS — Party win rate {:.1}% ≥ target {:.0}%",
            party_wr * 100.0,
            SONLT_TARGET_WIN_RATE * 100.0
        );
    } else {
        println!(
            "  ✗ FAIL — Party win rate {:.1}% < target {:.0}%",
            party_wr * 100.0,
            SONLT_TARGET_WIN_RATE * 100.0
        );
        println!();
        println!("  Possible causes:");
        println!("    - LoRA adapter trained on too few samples / epochs");
        println!("    - SON-LT warmup too short (mask too dense)");
        println!("    - Self-play data quality too low (HL vs HL may not be optimal)");
        println!("    - Token encoding loses critical tactical info (try richer state)");
    }
    println!();
    println!("═╡ Done ╞═");
}
