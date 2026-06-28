//! Convert binary postcard replay data to JSONL text format.
//!
//! The v2 replay generator (`bomber_06_replay_gen_v2`) writes binary postcard
//! records via `ReplayWriter` (layout: `len(4 LE) + postcard payload`). The
//! LoRA trainer (`riir_gpu::game::replay::parse_jsonl_dir`) expects JSONL text.
//! This converter bridges the gap.
//!
//! # Run
//!
//! ```sh
//! cargo run --example bomber_20_postcard_to_jsonl --features bomber -- \
//!     output/replays_v2/bomber_replay_v2_<timestamp>.jsonl \
//!     output/replays_v2/bomber_replay_v2_jsonl.jsonl
//! ```

use std::io::{BufWriter, Read, Write};
use std::path::PathBuf;

use katgpt_rs::pruners::bomber::replay::ReplaySample;

fn parse_args() -> (PathBuf, PathBuf) {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: {} <input_postcard.jsonl> <output_jsonl.jsonl>", args[0]);
        eprintln!();
        eprintln!("Converts binary postcard replay records to JSONL text format.");
        eprintln!("The LoRA trainer (parse_jsonl_dir) expects JSONL text, not binary.");
        std::process::exit(1);
    }
    (PathBuf::from(&args[1]), PathBuf::from(&args[2]))
}

fn main() {
    let (input_path, output_path) = parse_args();

    println!("╔═══ Postcard → JSONL Replay Converter ═══════════════════╗");
    println!("║  Input:  {}", input_path.display());
    println!("║  Output: {}", output_path.display());
    println!("╚══════════════════════════════════════════════════════════╝");

    let mut file = match std::fs::File::open(&input_path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("✗ Cannot open input: {e}");
            std::process::exit(1);
        }
    };

    let out_file = match std::fs::File::create(&output_path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("✗ Cannot create output: {e}");
            std::process::exit(1);
        }
    };
    let mut writer = BufWriter::new(out_file);

    // Get file size for progress reporting.
    let file_size = file.metadata().map(|m| m.len()).unwrap_or(0);
    let mut bytes_read = 0u64;
    let mut samples_written = 0u64;
    let mut samples_skipped = 0u64;
    let report_interval = 50_000u64;

    loop {
        // Read 4-byte LE length prefix.
        let mut len_buf = [0u8; 4];
        match file.read_exact(&mut len_buf) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => {
                eprintln!("✗ Read error at offset {bytes_read}: {e}");
                break;
            }
        }
        bytes_read += 4;

        let payload_len = u32::from_le_bytes(len_buf) as usize;
        if payload_len > 10 * 1024 * 1024 {
            eprintln!("✗ Implausible payload length {payload_len} at offset {bytes_read}");
            break;
        }

        // Read postcard payload.
        let mut payload = vec![0u8; payload_len];
        if let Err(e) = file.read_exact(&mut payload) {
            eprintln!("✗ Payload read error at offset {bytes_read}: {e}");
            break;
        }
        bytes_read += payload_len as u64;

        // Deserialize postcard → ReplaySample.
        let sample = match ReplaySample::from_bytes(&payload) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("  [skip] postcard deserialize error: {e}");
                samples_skipped += 1;
                continue;
            }
        };

        // Serialize as JSONL text line.
        let json = match serde_json::to_string(&sample) {
            Ok(j) => j,
            Err(e) => {
                eprintln!("  [skip] json serialize error: {e}");
                samples_skipped += 1;
                continue;
            }
        };

        if let Err(e) = writeln!(writer, "{json}") {
            eprintln!("✗ Write error: {e}");
            break;
        }
        samples_written += 1;

        if samples_written.is_multiple_of(report_interval) {
            let pct = if file_size > 0 {
                (bytes_read as f64 / file_size as f64) * 100.0
            } else {
                0.0
            };
            println!("  {samples_written} samples ({pct:.1}% of {file_size} bytes)");
        }
    }

    writer.flush().expect("flush output");
    println!();
    println!("═══════════════════════════════════════════════════════════");
    println!("  CONVERSION COMPLETE");
    println!("  Samples written: {samples_written}");
    println!("  Samples skipped: {samples_skipped}");
    println!("  Output:          {}", output_path.display());
    println!("═══════════════════════════════════════════════════════════");
}
