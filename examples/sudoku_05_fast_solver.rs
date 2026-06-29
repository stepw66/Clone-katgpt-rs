//! Quick sanity check for `Sudoku9x9::solve_fast()` (MRV + constraint propagation).
//! Run: cargo run --release --example sudoku_05_fast_solver --features sudoku

#![cfg(feature = "sudoku")]

use katgpt_percepta::Sudoku9x9;

fn main() {
    let mut b = Sudoku9x9::arto_inkala();
    let clues = b.clue_count();
    let (solved, steps) = b.solve_fast();
    println!("Arto Inkala: {clues} clues");
    println!("solve_fast: solved={solved}, steps={steps}, is_solved={}", b.is_solved());
    println!();
    print!("{}", b.display());
}
