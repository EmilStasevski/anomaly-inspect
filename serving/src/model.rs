//! Loading and running one exported model.

use std::fs;
use std::path::Path;
use std::sync::Mutex;

use anyhow::{Context, Result, anyhow};
use ndarray::{Array2, Array4, Ix2};
use ort::environment::Environment;
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::TensorRef;
use serde::Deserialize;

use crate::preprocess::Preprocess;

pub const MODEL_FILE: &str = "model.onnx";
pub const METADATA_FILE: &str = "model.json";

#[derive(Debug, Clone, Deserialize)]
pub struct Metadata {
    pub category: String,
    pub threshold: f64,
    pub heat_low: f32,
    pub heat_high: f32,
    pub preprocess: Preprocess,
}

pub struct Model {
    pub metadata: Metadata,
    // `Session::run` takes `&mut self`, so requests run one at a time per model. ONNX Runtime
    // parallelises inside each run, which is what the intra-op thread count controls.
    session: Mutex<Session>,
}

impl Model {
    pub fn load(environment: &Environment, directory: &Path, intra_threads: usize) -> Result<Self> {
        let metadata_path = directory.join(METADATA_FILE);
        let metadata: Metadata = serde_json::from_str(
            &fs::read_to_string(&metadata_path)
                .with_context(|| format!("reading {}", metadata_path.display()))?,
        )
        .with_context(|| format!("parsing {}", metadata_path.display()))?;

        let session = Session::builder(environment)
            .map_err(|error| anyhow!("{error}"))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|error| anyhow!("{error}"))?
            .with_intra_threads(intra_threads)
            .map_err(|error| anyhow!("{error}"))?
            .commit_from_file(directory.join(MODEL_FILE))
            .with_context(|| format!("loading the model in {}", directory.display()))?;

        Ok(Self {
            metadata,
            session: Mutex::new(session),
        })
    }

    /// Returns the image score and the anomaly map.
    pub fn infer(&self, tensor: &Array4<f32>) -> Result<(f32, Array2<f32>)> {
        let mut session = self
            .session
            .lock()
            .map_err(|_| anyhow!("a previous inference panicked"))?;
        let outputs = session.run(ort::inputs!["image" => TensorRef::from_array_view(tensor)?])?;
        let score = outputs["score"].try_extract_array::<f32>()?;
        let anomaly_map = outputs["anomaly_map"]
            .try_extract_array::<f32>()?
            .index_axis_move(ndarray::Axis(0), 0)
            .into_dimensionality::<Ix2>()?
            .to_owned();
        Ok((score[[0]], anomaly_map))
    }
}

/// The process-wide ONNX Runtime environment. Telemetry is switched off.
pub fn create_environment() -> Result<Environment> {
    ort::init()
        .with_name("anomaly-serving")
        .with_telemetry(false)
        .build()
        .map_err(|error| anyhow!("{error}"))
}

/// Version information of the linked ONNX Runtime, as reported by the `ort` crate.
pub fn runtime_info() -> &'static str {
    ort::info()
}
