//! Motor-Gated DEC Field — Example 2: motor gate channel isolation (Plan 357 T4.2).
//!
//! A 4-channel field with an identical bump on every channel. The motor gate
//! is applied to channels 0 and 1 only (motor_dim=2); channels 2 and 3
//! propagate freely. After a few ticks, channels 0/1 are amplified relative to
//! 2/3 — demonstrating the per-channel gain modulation (paper Experiment 3's
//! "body-selective motor channels").
//!
//! Run: `cargo run --example motor_gated_field_02_motor_gate --features motor_gated_field`

#![cfg(feature = "motor_gated_field")]

use katgpt_core::dec::{CellComplex, CochainField, evolve_motor_gated_field};

const W: usize = 16;
const H: usize = 16;
const DIM: usize = 4;
const MOTOR_DIM: usize = 2;

fn main() {
    println!("Plan 357 — Motor-Gated DEC Field, Example 2: motor gate isolation");
    println!("  16×16 grid, 4 channels, motor on channels 0..2, standard ReLU, dt=0.3");
    println!();

    let cx = CellComplex::grid_2d(W, H);

    // Two identical fields: one WITH motor, one WITHOUT (ballistic baseline).
    let make_field = || {
        let mut f = CochainField::zeros(0, cx.n_vertices(), DIM);
        for ch in 0..DIM {
            place_bump(&mut f, 8, 8, ch, 1.0, 2.0);
        }
        f
    };

    let mut with_motor = make_field();
    let mut no_motor = make_field();
    let mut lap1 = CochainField::zeros(0, cx.n_vertices(), DIM);
    let mut relu1 = CochainField::zeros(0, cx.n_vertices(), DIM);
    let mut lap2 = CochainField::zeros(0, cx.n_vertices(), DIM);
    let mut relu2 = CochainField::zeros(0, cx.n_vertices(), DIM);

    // Motor: channels 0,1 get gain (1 + dt*motor) = 1.45 per tick.
    let motor = [0.5f32, 0.5];
    let dt = 0.3;

    println!("Channel L1 mass over 5 ticks (motor gate amplifies channels 0,1):");
    println!(
        "{:>6}  {:>12}  {:>12}  {:>12}  {:>12}",
        "tick", "ch0(gated)", "ch1(gated)", "ch2(free)", "ch3(free)"
    );
    print_mass_row(0, &with_motor, &no_motor);

    for tick in 1..=5 {
        evolve_motor_gated_field(
            &cx,
            &mut with_motor,
            &motor,
            MOTOR_DIM,
            dt,
            0.0,
            &mut lap1,
            &mut relu1,
        );
        evolve_motor_gated_field(&cx, &mut no_motor, &[], 0, dt, 0.0, &mut lap2, &mut relu2);
        print_mass_row(tick, &with_motor, &no_motor);
    }

    println!();
    println!("The gated channels (0,1) diverge from the free channels (2,3):");
    println!("  ch0/ch2 ratio (with motor) grows each tick — the motor gain compounds.");
    println!("  ch2 and ch3 are identical between with-motor and no-motor runs (no leak).");
}

fn print_mass_row(tick: usize, with_motor: &CochainField, no_motor: &CochainField) {
    let m0 = channel_l1(with_motor, 0);
    let m1 = channel_l1(with_motor, 1);
    let m2_with = channel_l1(with_motor, 2);
    let m3_with = channel_l1(with_motor, 3);
    let m2_no = channel_l1(no_motor, 2);
    let m3_no = channel_l1(no_motor, 3);
    let leak2 = (m2_with - m2_no).abs();
    let leak3 = (m3_with - m3_no).abs();
    println!(
        "{tick:>6}  {m0:>12.4}  {m1:>12.4}  {m2_with:>12.4}  {m3_with:>12.4}  (leak ch2={leak2:.1e}, ch3={leak3:.1e})"
    );
}

fn channel_l1(field: &CochainField, ch: usize) -> f32 {
    let mut s = 0.0f32;
    for i in 0..W * H {
        s += field.data[i * DIM + ch].abs();
    }
    s
}

fn place_bump(field: &mut CochainField, cx: usize, cy: usize, ch: usize, amp: f32, sigma: f32) {
    for y in 0..H {
        for x in 0..W {
            let dx = x as f32 - cx as f32;
            let dy = y as f32 - cy as f32;
            let r2 = dx * dx + dy * dy;
            field.data[(y * W + x) * DIM + ch] = amp * (-r2 / (2.0 * sigma * sigma)).exp();
        }
    }
}
