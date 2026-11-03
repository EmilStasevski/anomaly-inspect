use std::collections::BTreeMap;
use std::env;
use std::io::{Read, Write};
use std::net::TcpStream;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

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

/// `anomaly-serving healthcheck`: the runtime image has no shell or curl for Docker to call.
fn healthcheck(address: &str) -> Result<()> {
    let port = address
        .rsplit(':')
        .next()
        .context("BIND_ADDRESS has no port")?;
    let mut stream = TcpStream::connect(("127.0.0.1", port.parse::<u16>()?))?;
    stream.set_read_timeout(Some(Duration::from_secs(4)))?;
    stream.write_all(b"GET /health HTTP/1.0\r\nHost: localhost\r\n\r\n")?;
    let mut response = String::new();
    stream.read_to_string(&mut response)?;
    if !response.starts_with("HTTP/1.0 200") && !response.starts_with("HTTP/1.1 200") {
        bail!(
            "unhealthy: {}",
            response.lines().next().unwrap_or("no response")
        );
    }
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    let address = env_or("BIND_ADDRESS", "127.0.0.1:8080");
    if env::args().nth(1).as_deref() == Some("healthcheck") {
        return healthcheck(&address);
    }

    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()))
        .init();

    let root = PathBuf::from(env_or("MODEL_ROOT", "models/onnx"));
    let only = env::var("MODEL_CATEGORIES").ok();
    let threads: usize = env_or("INTRA_THREADS", "4")
        .parse()
        .context("INTRA_THREADS must be an integer")?;

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
