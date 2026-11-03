//! HTTP interface: `POST /predict`, `GET /health`, `GET /metrics`.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Instant;

use axum::extract::{DefaultBodyLimit, Multipart, Query, State};
use axum::http::{StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use prometheus::{
    Encoder, HistogramOpts, HistogramVec, IntCounterVec, Opts, Registry, TextEncoder,
};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::heatmap;
use crate::model::Model;

const MAX_UPLOAD_BYTES: usize = 32 * 1024 * 1024;
const LATENCY_BUCKETS: &[f64] = &[
    0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0,
];

#[derive(Clone)]
pub struct Metrics {
    registry: Registry,
    predictions: IntCounterVec,
    errors: IntCounterVec,
    stage_seconds: HistogramVec,
}

impl Metrics {
    pub fn new() -> prometheus::Result<Self> {
        let registry = Registry::new();
        let predictions = IntCounterVec::new(
            Opts::new("predictions_total", "Completed predictions"),
            &["category", "decision"],
        )?;
        let errors = IntCounterVec::new(
            Opts::new("prediction_errors_total", "Rejected or failed requests"),
            &["reason"],
        )?;
        let stage_seconds = HistogramVec::new(
            HistogramOpts::new("prediction_stage_seconds", "Duration of each request stage")
                .buckets(LATENCY_BUCKETS.to_vec()),
            &["stage"],
        )?;
        registry.register(Box::new(predictions.clone()))?;
        registry.register(Box::new(errors.clone()))?;
        registry.register(Box::new(stage_seconds.clone()))?;
        Ok(Self {
            registry,
            predictions,
            errors,
            stage_seconds,
        })
    }
}

#[derive(Clone)]
pub struct AppState {
    pub models: Arc<BTreeMap<String, Arc<Model>>>,
    pub metrics: Metrics,
}

#[derive(Debug, Error)]
pub enum ApiError {
    #[error("{0}")]
    BadRequest(String),
    #[error("{0}")]
    UnknownCategory(String),
    #[error("inference failed")]
    Internal(#[from] anyhow::Error),
}

impl ApiError {
    fn reason(&self) -> &'static str {
        match self {
            Self::BadRequest(_) => "bad_request",
            Self::UnknownCategory(_) => "unknown_category",
            Self::Internal(_) => "internal",
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let status = match &self {
            Self::BadRequest(_) => StatusCode::BAD_REQUEST,
            Self::UnknownCategory(_) => StatusCode::NOT_FOUND,
            Self::Internal(error) => {
                tracing::error!("{error:#}");
                StatusCode::INTERNAL_SERVER_ERROR
            }
        };
        (
            status,
            Json(serde_json::json!({ "error": self.to_string() })),
        )
            .into_response()
    }
}

#[derive(Debug, Deserialize)]
pub struct PredictParams {
    category: Option<String>,
    #[serde(default)]
    heatmap: bool,
}

#[derive(Debug, Serialize)]
pub struct Prediction {
    category: String,
    score: f32,
    threshold: f64,
    is_anomalous: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    heatmap_png_base64: Option<String>,
}

fn select_model(state: &AppState, category: Option<&str>) -> Result<Arc<Model>, ApiError> {
    match category {
        Some(name) => state
            .models
            .get(name)
            .cloned()
            .ok_or_else(|| ApiError::UnknownCategory(format!("unknown category '{name}'"))),
        None if state.models.len() == 1 => Ok(state
            .models
            .values()
            .next()
            .expect("length checked")
            .clone()),
        None => Err(ApiError::BadRequest(
            "several models are loaded, so the 'category' query parameter is required".into(),
        )),
    }
}

async fn read_image(multipart: &mut Multipart) -> Result<axum::body::Bytes, ApiError> {
    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|error| ApiError::BadRequest(format!("invalid multipart body: {error}")))?
    {
        if field.name() == Some("image") {
            return field
                .bytes()
                .await
                .map_err(|error| ApiError::BadRequest(format!("could not read upload: {error}")));
        }
    }
    Err(ApiError::BadRequest(
        "missing multipart field 'image'".into(),
    ))
}

fn run_prediction(
    model: &Model,
    bytes: &[u8],
    with_heatmap: bool,
    metrics: &Metrics,
) -> Result<Prediction, ApiError> {
    let started = Instant::now();
    let image = image::load_from_memory(bytes)
        .map_err(|error| ApiError::BadRequest(format!("could not decode image: {error}")))?;
    let preprocess = &model.metadata.preprocess;
    let cropped = preprocess.crop_image(&image);
    let tensor = preprocess.tensor(&cropped);
    metrics
        .stage_seconds
        .with_label_values(&["preprocess"])
        .observe(started.elapsed().as_secs_f64());

    let started = Instant::now();
    let (score, anomaly_map) = model.infer(&tensor)?;
    metrics
        .stage_seconds
        .with_label_values(&["inference"])
        .observe(started.elapsed().as_secs_f64());

    let metadata = &model.metadata;
    let heatmap_png_base64 = if with_heatmap {
        let rendered = heatmap::render(
            &cropped,
            &anomaly_map,
            metadata.heat_low,
            metadata.heat_high,
        );
        Some(heatmap::to_base64_png(&rendered)?)
    } else {
        None
    };

    let threshold = metadata.threshold;
    let is_anomalous = f64::from(score) >= threshold;
    Ok(Prediction {
        category: metadata.category.clone(),
        score,
        threshold,
        is_anomalous,
        heatmap_png_base64,
    })
}

async fn predict(
    State(state): State<AppState>,
    Query(params): Query<PredictParams>,
    mut multipart: Multipart,
) -> Result<Json<Prediction>, ApiError> {
    let metrics = state.metrics.clone();
    let result = async {
        let model = select_model(&state, params.category.as_deref())?;
        let bytes = read_image(&mut multipart).await?;
        let worker_metrics = state.metrics.clone();
        tokio::task::spawn_blocking(move || {
            run_prediction(&model, &bytes, params.heatmap, &worker_metrics)
        })
        .await
        .map_err(|error| ApiError::Internal(error.into()))?
    }
    .await;

    match result {
        Ok(prediction) => {
            let decision = if prediction.is_anomalous {
                "defect"
            } else {
                "pass"
            };
            metrics
                .predictions
                .with_label_values(&[prediction.category.as_str(), decision])
                .inc();
            Ok(Json(prediction))
        }
        Err(error) => {
            metrics.errors.with_label_values(&[error.reason()]).inc();
            Err(error)
        }
    }
}

async fn health(State(state): State<AppState>) -> impl IntoResponse {
    let categories: Vec<&String> = state.models.keys().collect();
    let status = if categories.is_empty() {
        StatusCode::SERVICE_UNAVAILABLE
    } else {
        StatusCode::OK
    };
    let body = serde_json::json!({
        "status": if categories.is_empty() { "no models loaded" } else { "ok" },
        "categories": categories,
    });
    (status, Json(body))
}

async fn metrics(State(state): State<AppState>) -> Result<Response, ApiError> {
    let mut buffer = Vec::new();
    let encoder = TextEncoder::new();
    encoder
        .encode(&state.metrics.registry.gather(), &mut buffer)
        .map_err(|error| ApiError::Internal(error.into()))?;
    Ok((
        [(header::CONTENT_TYPE, encoder.format_type().to_string())],
        buffer,
    )
        .into_response())
}

pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/predict", post(predict))
        .route("/health", get(health))
        .route("/metrics", get(metrics))
        .layer(DefaultBodyLimit::max(MAX_UPLOAD_BYTES))
        .with_state(state)
}
