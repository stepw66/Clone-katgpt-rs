//! FFT self-play replay generation — outputs per-move JSONL samples for
//! riir-ai LoRA training (Plan 296 T7.3).
//!
//! Runs N FFT battles (HL vs HL by default; or any FftPlayer match-up) and
//! writes one `FftSample` JSON line per unit-turn. The schema matches
//! `riir_gpu::game::fft_replay::FftSample` exactly.
//!
//! ## Usage
//!
//! ```sh
//! # Default: 200 games, output to katgpt-rs/output/replays/fft/
//! cargo run --example fft_05_replay_gen --features fft --release -- \
//!     --games 200
//!
//! # Custom output path:
//! cargo run --example fft_05_replay_gen --features fft --release -- \
//!     --games 500 --output /tmp/fft_replays.jsonl
//! ```
//!
//! ## Output
//!
//! `output/replays/fft/fft_replay_<timestamp>.jsonl` — one JSON object per line.
//! Each line is a `FftSample` with 57-token battle state, action, quality, tick,
//! unit_id, team, and optional BLAKE3 checksum.

#![cfg(feature = "fft")]

use std::io::Write;
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fastrand::Rng;

use katgpt_rs::pruners::fft::players::HLFFTPlayer;
use katgpt_rs::pruners::fft::replay_encode::{FFT_STATE_LEN, encode_battle_state};
use katgpt_rs::pruners::fft::{Action, ActionType, BattleState, FftPlayer, TURN_LIMIT, Team};

// ── CLI ──────────────────────────────────────────────────────────

/// Sample quality threshold below which we drop the sample.
const DEFAULT_QUALITY_THRESHOLD: f32 = 0.5;

struct Config {
    games: usize,
    output: PathBuf,
    /// Minimum outcome quality (0=loss, 0.5=draw, 1=win) for samples to be
    /// written. Default 0.5 means only survivors / winners emit samples.
    quality_threshold: f32,
}

fn parse_args() -> Config {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut config = Config {
        games: 200,
        output: PathBuf::from("output/replays/fft"),
        quality_threshold: DEFAULT_QUALITY_THRESHOLD,
    };

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--games" | "-n" => {
                i += 1;
                if i < args.len() {
                    config.games = args[i].parse().unwrap_or_else(|_| {
                        eprintln!("Invalid --games value: {}", args[i]);
                        std::process::exit(1);
                    });
                }
            }
            "--output" | "-o" => {
                i += 1;
                if i < args.len() {
                    config.output = PathBuf::from(&args[i]);
                }
            }
            "--quality-threshold" | "-q" => {
                i += 1;
                if i < args.len() {
                    config.quality_threshold = args[i].parse().unwrap_or_else(|_| {
                        eprintln!("Invalid --quality-threshold: {}", args[i]);
                        std::process::exit(1);
                    });
                }
            }
            "--help" | "-h" => {
                eprintln!("FFT Self-Play Replay Generator (Plan 296 T7.3)");
                eprintln!();
                eprintln!("Options:");
                eprintln!("  --games, -n              Number of battles (default: 200)");
                eprintln!("  --output, -o             Output path:");
                eprintln!("                              dir   → <dir>/fft_replay_<ts>.jsonl");
                eprintln!("                              file  → write directly to file");
                eprintln!("  --quality-threshold, -q  Min quality to emit (default: 0.5)");
                std::process::exit(0);
            }
            other => {
                eprintln!("Unknown option: {other}. Use --help for usage.");
                std::process::exit(1);
            }
        }
        i += 1;
    }

    config
}

/// Resolve output path. If `raw` is a directory (or doesn't have a .jsonl
/// extension), append `fft_replay_<timestamp>.jsonl`.
fn resolve_output(raw: &PathBuf) -> PathBuf {
    if raw.extension().and_then(|e| e.to_str()) == Some("jsonl") {
        return raw.clone();
    }
    // Treat as directory.
    std::fs::create_dir_all(raw).ok();
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    raw.join(format!("fft_replay_{ts}.jsonl"))
}

// ── Sample Encoding ──────────────────────────────────────────────

/// Encode a single unit-turn decision as a JSONL line matching
/// `riir_gpu::game::fft_replay::FftSampleJson`.
fn encode_sample_jsonl(state: &BattleState, action: &Action, unit_id: u8, quality: f32) -> String {
    let mut state_tokens = [0u8; FFT_STATE_LEN];
    encode_battle_state(state, &mut state_tokens);

    let action_label = match action.action_type {
        ActionType::Attack => "Attack",
        ActionType::Defend => "Defend",
        ActionType::BlackMagic => "BlackMagic",
        ActionType::WhiteMagic => "WhiteMagic",
        ActionType::Potion => "Potion",
        ActionType::Wait => "Wait",
        ActionType::CurePoison => "CurePoison",
        ActionType::Esuna => "Esuna",
        ActionType::Dispel => "Dispel",
    };
    let team_label = if unit_id < 4 { "Party" } else { "Enemy" };

    let target_id = action
        .target_id
        .map(|t| t.to_string())
        .unwrap_or_else(|| "null".to_string());
    let move_to = action
        .move_to
        .map(|p| (p.y * 8 + p.x) as u8)
        .map(|v| v.to_string())
        .unwrap_or_else(|| "null".to_string());

    // Serialize the state array inline to avoid serde_json dep in the example.
    let state_str: String = state_tokens
        .iter()
        .map(|t| t.to_string())
        .collect::<Vec<_>>()
        .join(",");

    format!(
        r#"{{"state":[{state_str}],"unit_id":{unit_id},"action":{{"type":"{action_label}","target_id":{target_id},"move_to":{move_to}}},"quality":{quality},"tick":{tick},"team":"{team_label}"}}"#,
        tick = state.tick,
    )
}

// ── Instrumented Battle Runner ───────────────────────────────────

/// Like `run_fft_battle` but records every action choice into `samples`.
///
/// Returns the winning team (or `None` for timeout draw).
fn run_instrumented_battle(
    party: &mut [Box<dyn FftPlayer>],
    enemy: &mut [Box<dyn FftPlayer>],
    turn_limit: u32,
    rng: &mut Rng,
    samples: &mut Vec<String>,
    quality_threshold: f32,
) -> Option<Team> {
    let mut battle = BattleState::new();

    // Pre-compute the unit_id → team lookup (4 party + 4 enemy).
    let unit_teams: Vec<Team> = (0u8..8)
        .map(|id| {
            battle
                .units
                .get(id as usize)
                .map(|u| u.team)
                .unwrap_or(Team::Party)
        })
        .collect();

    // Collect per-unit samples during the battle keyed by unit_id.
    // After the battle, we know the winner and can assign quality.
    struct RecordedAction {
        jsonl: String,
        unit_id: u8,
    }
    let mut recorded: Vec<RecordedAction> = Vec::new();

    for _ in 0..turn_limit {
        battle.advance_ct();
        battle.tick_effects();

        let ready = battle.ready_units();
        if ready.is_empty() {
            continue;
        }

        // Collect actions.
        let mut actions: Vec<(u8, Action)> = Vec::with_capacity(8);
        for &unit_id in &ready {
            battle.units[unit_id as usize].defending = false;
            let unit = &battle.units[unit_id as usize];
            if !unit.alive {
                continue;
            }
            let player = match unit.team {
                Team::Party => &mut party[unit_id as usize],
                Team::Enemy => &mut enemy[(unit_id - 4) as usize],
            };
            let action = player.select_action(unit_id, &battle, rng);
            actions.push((unit_id, action));
        }

        // Record each action with the pre-resolution battle state.
        for (unit_id, action) in &actions {
            let jsonl = encode_sample_jsonl(&battle, action, *unit_id, -1.0);
            recorded.push(RecordedAction {
                jsonl,
                unit_id: *unit_id,
            });
        }

        // Resolve actions in order.
        for (unit_id, action) in &actions {
            katgpt_rs::pruners::fft::resolve_action(&mut battle, *unit_id, action, rng);
            if battle.check_winner().is_some() {
                break;
            }
        }

        battle.reset_ct(&ready);

        if battle.check_winner().is_some() {
            break;
        }
    }

    let winner = battle.check_winner().or_else(|| {
        let party_hp = battle.team_hp(Team::Party);
        let enemy_hp = battle.team_hp(Team::Enemy);
        match party_hp.cmp(&enemy_hp) {
            std::cmp::Ordering::Greater => Some(Team::Party),
            std::cmp::Ordering::Less => Some(Team::Enemy),
            std::cmp::Ordering::Equal => None,
        }
    });

    // Assign quality per sample based on the unit's team outcome.
    for rec in recorded {
        let team = unit_teams[rec.unit_id as usize];
        let quality = match winner {
            Some(t) if t == team => 1.0,
            Some(_) => 0.0,
            None => 0.5,
        };
        if quality >= quality_threshold {
            // Patch the quality value into the pre-encoded JSONL line.
            // We encoded quality=-1.0 above as a placeholder; rewrite it now.
            let patched = rec
                .jsonl
                .replace(r#""quality":-1"#, &format!("\"quality\":{quality}"));
            samples.push(patched);
        }
    }

    winner
}

// ── Main ─────────────────────────────────────────────────────────

fn main() {
    let config = parse_args();
    let output_path = resolve_output(&config.output);

    eprintln!("=== FFT Self-Play Replay Generator (Plan 296 T7.3) ===");
    eprintln!(
        "Games: {}, Quality threshold: {}, Output: {}",
        config.games,
        config.quality_threshold,
        output_path.display()
    );

    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent).ok();
    }

    let mut file = match std::fs::File::create(&output_path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!(
                "Failed to create output file {}: {e}",
                output_path.display()
            );
            std::process::exit(1);
        }
    };

    let mut rng = Rng::with_seed(42);
    let start = Instant::now();

    let mut party_wins = 0usize;
    let mut enemy_wins = 0usize;
    let mut draws = 0usize;
    let mut total_samples = 0usize;

    for game_num in 1..=config.games {
        // Each game uses a fresh party / enemy (HL players carry state between
        // games — that's the intended learning signal).
        let mut party: Vec<Box<dyn FftPlayer>> = vec![
            Box::new(HLFFTPlayer::new()),
            Box::new(HLFFTPlayer::new()),
            Box::new(HLFFTPlayer::new()),
            Box::new(HLFFTPlayer::new()),
        ];
        let mut enemy: Vec<Box<dyn FftPlayer>> = vec![
            Box::new(HLFFTPlayer::new()),
            Box::new(HLFFTPlayer::new()),
            Box::new(HLFFTPlayer::new()),
            Box::new(HLFFTPlayer::new()),
        ];

        let mut samples: Vec<String> = Vec::new();
        let winner = run_instrumented_battle(
            &mut party,
            &mut enemy,
            TURN_LIMIT,
            &mut rng,
            &mut samples,
            config.quality_threshold,
        );

        match winner {
            Some(Team::Party) => party_wins += 1,
            Some(Team::Enemy) => enemy_wins += 1,
            None => draws += 1,
        }

        for line in &samples {
            if writeln!(file, "{line}").is_err() {
                eprintln!("[ERROR] write failed on game {game_num}");
                break;
            }
        }
        total_samples += samples.len();

        if game_num % 10 == 0 || game_num == config.games {
            let elapsed = start.elapsed().as_secs_f32();
            let gps = game_num as f32 / elapsed.max(0.001);
            eprintln!(
                "  [{game_num:>4}/{total}] P:{party_wins} E:{enemy_wins} D:{draws} | samples:{total_samples} | {gps:.1} games/s",
                total = config.games,
            );
        }
    }

    drop(file);

    let file_size = std::fs::metadata(&output_path)
        .map(|m| m.len())
        .unwrap_or(0);

    eprintln!();
    eprintln!("=== Results ===");
    eprintln!("Output:    {}", output_path.display());
    eprintln!("File size: {:.2} MB", file_size as f64 / 1e6);
    eprintln!(
        "Games:     {} ({:.1} games/s)",
        config.games,
        config.games as f64 / start.elapsed().as_secs_f64().max(0.001)
    );
    eprintln!(
        "Party wins: {party_wins} ({:.1}%)",
        party_wins as f64 / config.games as f64 * 100.0
    );
    eprintln!(
        "Enemy wins: {enemy_wins} ({:.1}%)",
        enemy_wins as f64 / config.games as f64 * 100.0
    );
    eprintln!("Draws:     {draws}");
    eprintln!(
        "Total samples: {total_samples} (quality >= {})",
        config.quality_threshold
    );
    eprintln!("Elapsed:   {:.2}s", start.elapsed().as_secs_f32());
}
