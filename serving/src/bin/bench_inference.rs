//! In-process latency: no HTTP, single image, after warm-up. Prints one JSON object.

use std::env;
use std::path::PathBuf;
use std::time::Instant;

use anomaly_serving::model::{Model, create_environment, runtime_info};
use anyhow::{Context, Result, bail};
use serde_json::json;

fn percentile(sorted: &[f64], fraction: f64) -> f64 {
    let rank = (fraction * (sorted.len() - 1) as f64).round() as usize;
    sorted[rank]
}

fn main() -> Result<()> {
    let args: Vec<String> = env::args().skip(1).collect();
    let [model_dir, image_path, threads, warmup, iterations] = args.as_slice() else {
        bail!("usage: bench_inference <model dir> <image> <intra threads> <warm-up> <iterations>");
    };
    let threads: usize = threads.parse().context("threads")?;
    let warmup: usize = warmup.parse().context("warm-up")?;
    let iterations: usize = iterations.parse().context("iterations")?;

    let load_start = Instant::now();
    let environment = create_environment()?;
    let model = Model::load(&environment, &PathBuf::from(model_dir), threads)?;
    let load_seconds = load_start.elapsed().as_secs_f64();

    let preprocess = &model.metadata.preprocess;
    let image = image::open(image_path).with_context(|| format!("opening {image_path}"))?;

    let preprocess_start = Instant::now();
    let tensor = preprocess.tensor(&preprocess.crop_image(&image));
    let preprocess_ms = preprocess_start.elapsed().as_secs_f64() * 1e3;

    for _ in 0..warmup {
        model.infer(&tensor)?;
    }

    let mut samples: Vec<f64> = Vec::with_capacity(iterations);
    let total_start = Instant::now();
    for _ in 0..iterations {
        let start = Instant::now();
        std::hint::black_box(model.infer(&tensor)?);
        samples.push(start.elapsed().as_secs_f64() * 1e3);
    }
    let total_seconds = total_start.elapsed().as_secs_f64();
    samples.sort_by(f64::total_cmp);

    println!(
        "{}",
        json!({
            "category": model.metadata.category,
            "threads": threads,
            "warmup": warmup,
            "iterations": iterations,
            "load_seconds": load_seconds,
            "preprocess_ms": preprocess_ms,
            "p50_ms": percentile(&samples, 0.50),
            "p95_ms": percentile(&samples, 0.95),
            "p99_ms": percentile(&samples, 0.99),
            "throughput_per_s": iterations as f64 / total_seconds,
            "runtime": runtime_info(),
        })
    );
    Ok(())
}
