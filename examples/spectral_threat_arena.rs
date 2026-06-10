//! Spectral Threat Arena — GOAT Gate for Plan 241.
//!
//! Compares reactive NPC vs spectral NPC in a scripted 3-hit combo scenario.
//! The spectral NPC uses LinOSS modal analysis to predict combo timing.
//!
//! Run: `cargo run --example spectral_threat_arena --features spectral_threat`

use katgpt_core::sense::CombatRhythmTracker;

// ── Constants ──────────────────────────────────────────────────

/// Game tick interval in seconds (16ms = ~62.5 Hz).
const DT: f32 = 0.016;
/// Ticks per second.
const TPS: u32 = 62;
/// Total duration: 60 seconds.
const DURATION_SECS: u32 = 60;
/// Total ticks in the simulation.
const TOTAL_TICKS: u32 = TPS * DURATION_SECS;
/// Max NPC HP.
const MAX_HP: i32 = 1000;
/// Damage values for the 3-hit combo: slash, slash, heavy.
const COMBO_DAMAGE: [i32; 3] = [80, 80, 150];
/// Ticks between combo hits.
const TICKS_BETWEEN_HITS: [u32; 3] = [15, 15, 25]; // 240ms, 240ms, 400ms — fast combo
/// How many ticks before impact the threat becomes "visible" to reactive NPC.
const THREAT_VISIBILITY_TICKS: u32 = 8;
/// Dodge cooldown: NPC can't dodge again for N ticks after dodging.
const DODGE_COOLDOWN_TICKS: u32 = 25;

// ── NPC State ──────────────────────────────────────────────────

#[derive(Debug)]
struct NpcState {
    hp: i32,
    dodges: u32,
    hits_taken: u32,
    last_dodge_tick: u32,
    total_attacks: u32,
}

impl NpcState {
    fn new() -> Self {
        Self {
            hp: MAX_HP,
            dodges: 0,
            hits_taken: 0,
            last_dodge_tick: 0,
            total_attacks: 0,
        }
    }

    fn dodge_rate(&self) -> f32 {
        if self.total_attacks == 0 {
            return 0.0;
        }
        self.dodges as f32 / self.total_attacks as f32
    }

    fn hp_remaining_ratio(&self) -> f32 {
        (self.hp as f32 / MAX_HP as f32).max(0.0)
    }
}

// ── Attack Schedule ────────────────────────────────────────────

struct Attack {
    threat_tick: u32,
    impact_tick: u32,
    damage: i32,
    combo_index: usize, // 0, 1, or 2 — position in the 3-hit combo
}

struct AttackSchedule {
    attacks: Vec<Attack>,
}

impl AttackSchedule {
    fn generate() -> Self {
        let mut attacks = Vec::with_capacity(300);
        let mut tick = 50u32;

        while tick < TOTAL_TICKS {
            for i in 0..3 {
                if tick >= TOTAL_TICKS {
                    break;
                }
                let threat_tick = tick.saturating_sub(THREAT_VISIBILITY_TICKS);
                attacks.push(Attack {
                    threat_tick,
                    impact_tick: tick,
                    damage: COMBO_DAMAGE[i],
                    combo_index: i,
                });
                tick += TICKS_BETWEEN_HITS[i];
            }
        }
        Self { attacks }
    }
}

// ── Reactive NPC (NPC A) ──────────────────────────────────────

/// Reactive NPC: can only dodge when the threat becomes visible
/// (THREAT_VISIBILITY_TICKS before impact). If on cooldown, takes the hit.
fn run_reactive_npc(schedule: &AttackSchedule) -> NpcState {
    let mut npc = NpcState::new();

    for attack in &schedule.attacks {
        npc.total_attacks += 1;

        // Reactive: can only dodge when threat is visible AND cooldown allows
        let off_cooldown = attack.threat_tick > npc.last_dodge_tick + DODGE_COOLDOWN_TICKS;
        if off_cooldown {
            npc.dodges += 1;
            npc.last_dodge_tick = attack.threat_tick;
        } else {
            npc.hp = (npc.hp - attack.damage).max(0);
            npc.hits_taken += 1;
        }
    }

    npc
}

// ── Spectral NPC (NPC B) ──────────────────────────────────────

/// Spectral NPC: uses CombatRhythmTracker to predict combo timing.
///
/// Key advantage: the tracker learns the combo rhythm from damage events.
/// After seeing the combo pattern, it can predict when the next hit is coming
/// based on the oscillator's phase. The NPC can "pre-dodge" — use its cooldown
/// window BEFORE the threat becomes visible, if spectral features indicate
/// an imminent hit.
fn run_spectral_npc(schedule: &AttackSchedule) -> NpcState {
    let mut npc = NpcState::new();
    let mut tracker = CombatRhythmTracker::with_combat_frequencies(DT);
    tracker.register(1);

    // Pre-advance tracker state to simulate real-time physics.
    // In real game, the tracker runs every tick. We do the same but only
    // ingest non-zero damage events (zero forcing is natural oscillation decay).
    // We need to advance state between events to capture phase evolution.

    let mut attack_idx = 0usize;
    let mut pre_dodged_next: bool = false;

    for tick in 0..TOTAL_TICKS {
        // Check if an attack impacts this tick
        if attack_idx < schedule.attacks.len() && schedule.attacks[attack_idx].impact_tick == tick {
            let attack = &schedule.attacks[attack_idx];
            npc.total_attacks += 1;

            if pre_dodged_next {
                // We already committed to dodging this attack via spectral prediction
                npc.dodges += 1;
                npc.last_dodge_tick = tick;
                pre_dodged_next = false;
            } else {
                // Fall back to reactive: can we dodge?
                let off_cooldown = tick > npc.last_dodge_tick + DODGE_COOLDOWN_TICKS;
                let threat_visible = tick >= attack.threat_tick;
                if off_cooldown && threat_visible {
                    npc.dodges += 1;
                    npc.last_dodge_tick = tick;
                } else {
                    npc.hp = (npc.hp - attack.damage).max(0);
                    npc.hits_taken += 1;
                }
            }

            // Feed the damage impulse to tracker
            tracker.ingest_damage(1, attack.damage as f32, tick);
            attack_idx += 1;
        } else {
            // Advance tracker with zero forcing (natural oscillation decay)
            tracker.ingest_damage(1, 0.0, tick);
        }

        // Spectral prediction: look ahead to the next attack
        if !pre_dodged_next && attack_idx < schedule.attacks.len() {
            let next = &schedule.attacks[attack_idx];
            let can_dodge = tick > npc.last_dodge_tick + DODGE_COOLDOWN_TICKS;

            // Only consider pre-dodging if:
            // 1. Not on cooldown
            // 2. The next attack is approaching (within prediction window)
            // 3. The threat isn't visible yet (pure prediction zone)
            let prediction_window = THREAT_VISIBILITY_TICKS * 2; // Look ahead 2x the visibility
            let approaching = tick + prediction_window >= next.impact_tick;
            let not_visible_yet = tick < next.threat_tick;

            if can_dodge && approaching && not_visible_yet {
                let features = tracker.extract_features(1);

                // Debug: always print when we have ANY confidence
                if features.rhythm_confidence > 0.01 {
                    eprintln!(
                        "  [tick={tick}] conf={:.4} urgency={:.4} phase={:.4} freq={:.2} decay={:.2} event_count=nan",
                        features.rhythm_confidence,
                        features.dodge_urgency(),
                        features.vulnerability_phase,
                        features.combo_frequency,
                        features.burst_decay
                    );
                }

                // Only use spectral prediction after the tracker has learned the rhythm
                if features.rhythm_confidence > 0.3 {
                    let urgency = features.dodge_urgency();

                    // Pre-dodge if urgency indicates imminent damage peak
                    if urgency > 0.6 {
                        pre_dodged_next = true;
                        npc.last_dodge_tick = tick;
                        eprintln!("    → PRE-DODGE triggered at tick {tick}!");
                    }
                }
            }
        }
    }

    npc
}

// ── Main ───────────────────────────────────────────────────────

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Plan 241: Spectral Threat Arena — GOAT Gate               ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let schedule = AttackSchedule::generate();
    println!(
        "Attack schedule: {} attacks over {}s",
        schedule.attacks.len(),
        DURATION_SECS
    );
    println!(
        "Combo pattern: slash({}dmg) → slash({}dmg) → heavy({}dmg) @ fast cycle",
        COMBO_DAMAGE[0], COMBO_DAMAGE[1], COMBO_DAMAGE[2]
    );
    println!(
        "  Tick intervals: {}+{}+{} = {} ticks per combo cycle",
        TICKS_BETWEEN_HITS[0],
        TICKS_BETWEEN_HITS[1],
        TICKS_BETWEEN_HITS[2],
        TICKS_BETWEEN_HITS.iter().sum::<u32>()
    );
    println!(
        "  Threat visibility: {} ticks ({:.0}ms) before impact",
        THREAT_VISIBILITY_TICKS,
        THREAT_VISIBILITY_TICKS as f32 * DT * 1000.0
    );
    println!(
        "  Dodge cooldown: {} ticks ({:.0}ms)",
        DODGE_COOLDOWN_TICKS,
        DODGE_COOLDOWN_TICKS as f32 * DT * 1000.0
    );
    println!();

    // ── Run NPCs ───────────────────────────────────────────────
    println!("Running NPC A (reactive-only)...");
    let npc_a = run_reactive_npc(&schedule);

    println!("Running NPC B (spectral prediction)...");
    let npc_b = run_spectral_npc(&schedule);

    // ── Results ────────────────────────────────────────────────
    println!();
    println!("┌──────────────────────────────────────────────────────────┐");
    println!("│  RESULTS                                                 │");
    println!("├────────────────────┬─────────────┬─────────────┬─────────┤");
    println!("│  Metric            │  NPC A (R)  │  NPC B (S)  │  Δ      │");
    println!("├────────────────────┼─────────────┼─────────────┼─────────┤");
    println!(
        "│  Dodge Rate        │  {:5.1}%     │  {:5.1}%     │  {:+.1}%  │",
        npc_a.dodge_rate() * 100.0,
        npc_b.dodge_rate() * 100.0,
        (npc_b.dodge_rate() - npc_a.dodge_rate()) * 100.0
    );
    println!(
        "│  HP Remaining      │  {:5}       │  {:5}       │  {:+5}   │",
        npc_a.hp,
        npc_b.hp,
        npc_b.hp - npc_a.hp
    );
    println!(
        "│  HP Ratio          │  {:5.1}%     │  {:5.1}%     │  {:+.1}%  │",
        npc_a.hp_remaining_ratio() * 100.0,
        npc_b.hp_remaining_ratio() * 100.0,
        (npc_b.hp_remaining_ratio() - npc_a.hp_remaining_ratio()) * 100.0
    );
    println!(
        "│  Hits Taken        │  {:5}       │  {:5}       │  {:+5}   │",
        npc_a.hits_taken,
        npc_b.hits_taken,
        npc_b.hits_taken as i64 - npc_a.hits_taken as i64
    );
    println!(
        "│  Total Attacks     │  {:5}       │  {:5}       │         │",
        npc_a.total_attacks, npc_b.total_attacks
    );
    println!("└────────────────────┴─────────────┴─────────────┴─────────┘");
    println!();

    // ── GOAT Gate Verdict ──────────────────────────────────────
    println!("─── GOAT Gate ─────────────────────────────────────────────");

    let dodge_improvement = if npc_a.dodge_rate() > 0.001 {
        (npc_b.dodge_rate() - npc_a.dodge_rate()) / npc_a.dodge_rate() * 100.0
    } else if npc_b.dodge_rate() > 0.0 {
        f32::INFINITY
    } else {
        0.0
    };
    let hp_improvement = if npc_a.hp_remaining_ratio() > 0.001 {
        (npc_b.hp_remaining_ratio() - npc_a.hp_remaining_ratio()) / npc_a.hp_remaining_ratio()
            * 100.0
    } else if npc_b.hp_remaining_ratio() > 0.0 {
        f32::INFINITY
    } else {
        0.0
    };

    let dodge_pass = dodge_improvement >= 30.0;
    let hp_pass = hp_improvement >= 15.0;

    println!(
        "  Dodge improvement:  {:+.1}% {} (threshold: +30%)",
        dodge_improvement,
        if dodge_pass { "✅ PASS" } else { "❌ FAIL" }
    );
    println!(
        "  HP improvement:     {:+.1}% {} (threshold: +15%)",
        hp_improvement,
        if hp_pass { "✅ PASS" } else { "❌ FAIL" }
    );

    if dodge_pass && hp_pass {
        println!();
        println!("  🐐 GOAT CONFIRMED — spectral_threat passes arena proof.");
        println!("  Proposal: promote spectral_threat to default-ON.");
    } else if dodge_pass {
        println!();
        println!("  ⚠️  PARTIAL PASS — dodge rate met but HP improvement insufficient.");
        println!("  Recommendation: tune spectral weights, re-bench.");
    } else {
        println!();
        println!("  ❌ DEMOTE — spectral_threat fails GOAT gate.");
        println!("  Recommendation: keep as opt-in experiment, investigate frequency tuning.");
    }

    println!("─────────────────────────────────────────────────────────────");
}
