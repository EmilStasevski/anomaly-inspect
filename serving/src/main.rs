use std::collections::BTreeMap;
use std::env;
use std::path::PathBuf;
use std::sync::Arc;

use anomaly_serving::model::{MODEL_FILE, Model, create_environment, runtime_info};
use anomaly_serving::server::{AppState, Metrics, router};
use anyhow::{Context, Result, bail};
use tracing_subscriber::EnvFilter;

fn env_or(name: &str, default: &str) -> String {
    env::var(name).unwrap_or_else(|_| default.to_string())
}

fn load_models(
    root: &PathBuf,
    only: Option<&str>,
    threads: usize,
) -> Result<BTreeMap<String, Arc<Model>>> {
    let wanted: Option<Vec<&str>> = only.map(|list| list.split(',').map(str::trim).collect());
    let environment = create_environment()?;
    let mut models = BTreeMap::new();
    for entry in std::fs::read_dir(root).with_context(|| format!("reading {}", root.display()))? {
        let directory = entry?.path();
        let name = directory
            .file_name()
            .map(|n| n.to_string_lossy().into_owned());
        let Some(name) = name else { continue };
        if !directory.join(MODEL_FILE).is_file() {
            continue;
        }
        if wanted
            .as_ref()
            .is_some_and(|list| !list.contains(&name.as_str()))
        {
            continue;
        }
        tracing::info!(category = %name, threads, "loading model");
        models.insert(
            name,
            Arc::new(Model::load(&environment, &directory, threads)?),
        );
    }
    if models.is_empty() {
        bail!("no models found in {}", root.display());
    }
    Ok(models)
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()))
        .init();

    let root = PathBuf::from(env_or("MODEL_ROOT", "models/onnx"));
    let only = env::var("MODEL_CATEGORIES").ok();
    let threads: usize = env_or("INTRA_THREADS", "4")
        .parse()
        .context("INTRA_THREADS must be an integer")?;
    let address = env_or("BIND_ADDRESS", "127.0.0.1:8080");

    tracing::info!(runtime = runtime_info(), "starting");
    let models = load_models(&root, only.as_deref(), threads)?;
    let state = AppState {
        models: Arc::new(models),
        metrics: Metrics::new()?,
    };

    let listener = tokio::net::TcpListener::bind(&address)
        .await
        .with_context(|| format!("binding {address}"))?;
    tracing::info!(%address, "listening");
    axum::serve(listener, router(state))
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await?;
    Ok(())
}
