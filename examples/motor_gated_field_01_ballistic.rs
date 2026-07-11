//! Motor-Gated DEC Field — Example 1: ballistic bump propagation (Plan 357 T4.1).
//!
//! A single positive bump on a 32×32 grid, no motor gate (`motor_dim=0`),
//! standard ReLU. The bump spreads locally under the Hodge-Laplacian lateral
//! propagation — the DEC `d∘d=0` identity makes teleporting structurally
//! impossible (paper Experiment 1's "0.0% teleportation" result).
//!
//! ASCII-art visualization of channel 0 every 10 ticks shows the bump
//! diffusing outward without jumping.
//!
//! Run: `cargo run --example motor_gated_field_01_ballistic --features motor_gated_field`

#![cfg(feature = "motor_gated_field")]

use katgpt_core::dec::{CellComplex, CochainField, evolve_motor_gated_field};

const W: usize = 32;
const H: usize = 32;
const DIM: usize = 1;

fn main() {
    println!("Plan 357 — Motor-Gated DEC Field, Example 1: ballistic propagation");
    println!("  32×32 grid, 1 channel, no motor gate, standard ReLU, dt=0.1");
    println!();

    let cx = CellComplex::grid_2d(W, H);
    let mut field = CochainField::zeros(0, cx.n_vertices(), DIM);
    place_bump(&mut field, 16, 16, 1.0, 2.0);

    let mut lap = CochainField::zeros(0, cx.n_vertices(), DIM);
    let mut relu = CochainField::zeros(0, cx.n_vertices(), DIM);

    print_field(&field, "tick 0");
    for tick in 1..=30 {
        evolve_motor_gated_field(&cx, &mut field, &[], 0, 0.1, 0.0, &mut lap, &mut relu);
        if tick % 10 == 0 {
            print_field(&field, &format!("tick {tick}"));
        }
    }

    let (cx_f, cy_f) = centroid(&field, 0);
    println!("Final centroid: ({cx_f:.2}, {cy_f:.2}) — started at (16.0, 16.0)");
    println!(
        "Centroid drift: {:.3} cells (G1 gate: ≤ 2.0)",
        ((cx_f - 16.0).powi(2) + (cy_f - 16.0).powi(2)).sqrt()
    );
}

fn place_bump(field: &mut CochainField, cx: usize, cy: usize, amp: f32, sigma: f32) {
    for y in 0..H {
        for x in 0..W {
            let dx = x as f32 - cx as f32;
            let dy = y as f32 - cy as f32;
            let r2 = dx * dx + dy * dy;
            field.data[y * W + x] = amp * (-r2 / (2.0 * sigma * sigma)).exp();
        }
    }
}

fn print_field(field: &CochainField, label: &str) {
    println!("── {label} ──");
    let max = field.data.iter().cloned().fold(0.0f32, f32::max).max(1e-9);
    let ramp = [' ', '.', ':', '-', '=', '+', '*', '#', '%', '@'];
    for y in 0..H {
        let mut row = String::with_capacity(W);
        for x in 0..W {
            let v = field.data[y * W + x].max(0.0) / max;
            let idx = (v * (ramp.len() - 1) as f32) as usize;
            row.push(ramp[idx.min(ramp.len() - 1)]);
        }
        println!("  {row}");
    }
    println!();
}

fn centroid(field: &CochainField, ch: usize) -> (f32, f32) {
    let mut sx = 0.0f32;
    let mut sy = 0.0f32;
    let mut mass = 0.0f32;
    for y in 0..H {
        for x in 0..W {
            let v = field.data[(y * W + x) * DIM + ch].abs();
            sx += x as f32 * v;
            sy += y as f32 * v;
            mass += v;
        }
    }
    if mass > 0.0 {
        (sx / mass, sy / mass)
    } else {
        (0.0, 0.0)
    }
}
