//! Compares two commits' worth of rows in `benches/history.csv`, joining on
//! everything except `time_ms` so it never compares across different
//! targets/thread-counts by accident.
//!
//! ```bash
//! cargo run --release --example bench_compare              # last two distinct commits
//! cargo run --release --example bench_compare <old> <new>  # specific commits (prefix match ok)
//! ```

use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct Key {
    os: String,
    arch: String,
    threads: u32,
    variant: String,
    causal: bool,
    seq_len: u32,
    d_head: u32,
}

struct Row {
    commit: String,
    key: Key,
    time_ms: f64,
}

fn parse_csv(path: &std::path::Path) -> Vec<Row> {
    let content = fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("couldn't read {}: {e}", path.display()));
    let mut out = Vec::new();
    for line in content.lines() {
        // Header line(s) — CI concatenates one CSV output per OS/arch leg,
        // each with its own header, so more than one can appear.
        if line.is_empty() || line.starts_with("timestamp,") {
            continue;
        }
        let f: Vec<&str> = line.split(',').collect();
        if f.len() != 10 {
            eprintln!("skipping malformed row: {line}");
            continue;
        }
        let (Ok(threads), Ok(causal), Ok(seq_len), Ok(d_head), Ok(time_ms)) = (
            f[4].parse::<u32>(),
            f[6].parse::<bool>(),
            f[7].parse::<u32>(),
            f[8].parse::<u32>(),
            f[9].parse::<f64>(),
        ) else {
            eprintln!("skipping malformed row: {line}");
            continue;
        };
        out.push(Row {
            commit: f[1].to_string(),
            key: Key {
                os: f[2].to_string(),
                arch: f[3].to_string(),
                threads,
                variant: f[5].to_string(),
                causal,
                seq_len,
                d_head,
            },
            time_ms,
        });
    }
    out
}

fn build_map(rows: &[Row], commit: &str) -> HashMap<Key, f64> {
    rows.iter()
        .filter(|r| r.commit == commit)
        .map(|r| (r.key.clone(), r.time_ms))
        .collect()
}

fn main() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("benches/history.csv");
    let rows = parse_csv(&path);
    if rows.is_empty() {
        eprintln!("no usable rows in {}", path.display());
        std::process::exit(1);
    }

    // Distinct commits in the order they first appear — since new
    // measurements are always appended, this is chronological.
    let mut commits: Vec<String> = Vec::new();
    for r in &rows {
        if !commits.contains(&r.commit) {
            commits.push(r.commit.clone());
        }
    }

    let args: Vec<String> = std::env::args().skip(1).collect();
    let (old_commit, new_commit) = if args.len() == 2 {
        let resolve = |arg: &str| -> String {
            commits
                .iter()
                .find(|c| c.as_str() == arg || c.starts_with(arg))
                .cloned()
                .unwrap_or_else(|| {
                    panic!("no data for commit matching {arg:?}; known commits: {commits:?}")
                })
        };
        (resolve(&args[0]), resolve(&args[1]))
    } else if commits.len() >= 2 {
        (
            commits[commits.len() - 2].clone(),
            commits[commits.len() - 1].clone(),
        )
    } else {
        println!(
            "only one commit ({}) in {} so far — nothing to compare yet. \
             Run `bench_quick --csv` again after a change and append it to add a second one.",
            commits[0],
            path.display()
        );
        return;
    };

    println!("comparing {old_commit} -> {new_commit}\n");

    let old_map = build_map(&rows, &old_commit);
    let new_map = build_map(&rows, &new_commit);

    let mut keys: Vec<Key> = new_map.keys().chain(old_map.keys()).cloned().collect();
    keys.sort();
    keys.dedup();

    println!(
        "{:<10} {:>5} {:>3} {:>6} {:>5} {:>6} {:>6} | {:>10} {:>10} {:>8}",
        "target", "thr", "var", "causal", "seq", "d", "", "old_ms", "new_ms", "change"
    );
    let mut only_old: Vec<&Key> = Vec::new();
    let mut only_new: Vec<&Key> = Vec::new();
    for key in &keys {
        let target = format!("{}/{}", key.os, key.arch);
        match (old_map.get(key), new_map.get(key)) {
            (Some(&old_t), Some(&new_t)) => {
                let pct = (new_t - old_t) / old_t * 100.0;
                let marker = if pct <= -3.0 {
                    "faster"
                } else if pct >= 3.0 {
                    "SLOWER"
                } else {
                    "~"
                };
                println!(
                    "{target:<10} {:>5} {:>3} {:>6} {:>5} {:>6} {:>6} | {old_t:>10.4} {new_t:>10.4} {pct:>+7.1}% {marker}",
                    key.threads, key.variant, key.causal, key.seq_len, key.d_head, "",
                );
            }
            (Some(_), None) => only_old.push(key),
            (None, Some(&new_t)) => {
                only_new.push(key);
                println!(
                    "{target:<10} {:>5} {:>3} {:>6} {:>5} {:>6} {:>6} | {:>10} {new_t:>10.4} {:>9}",
                    key.threads, key.variant, key.causal, key.seq_len, key.d_head, "", "-", "new",
                );
            }
            (None, None) => unreachable!(),
        }
    }

    if !only_old.is_empty() {
        println!(
            "\n{} measurement(s) present only in {old_commit}, not in {new_commit} (target/shape not re-measured)",
            only_old.len()
        );
    }
    if !only_new.is_empty() {
        println!(
            "{} measurement(s) present only in {new_commit}, not in {old_commit} (new target/shape, printed above as \"new\")",
            only_new.len()
        );
    }
}
