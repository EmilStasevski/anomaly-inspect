//! Write the preprocessed tensor as little-endian f32, for parity tests against Python.

use std::env;
use std::fs;

use anomaly_serving::preprocess::Preprocess;
use anyhow::{Context, Result, bail};

fn main() -> Result<()> {
    let args: Vec<String> = env::args().skip(1).collect();
    let [input, output] = args.as_slice() else {
        bail!("usage: preprocess_dump <input image> <output.bin>");
    };
    let preprocess = Preprocess::default();
    let image = image::open(input).with_context(|| format!("opening {input}"))?;
    let tensor = preprocess.tensor(&preprocess.crop_image(&image));
    let bytes: Vec<u8> = tensor
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    fs::write(output, bytes).with_context(|| format!("writing {output}"))?;
    Ok(())
}
