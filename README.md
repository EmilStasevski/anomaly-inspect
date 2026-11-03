# Visual quality inspection with anomaly detection

Detects and localises surface and structural defects in industrial parts, trained **only on images of good parts**. The project compares a simple baseline with PatchCore, tracks every run in MLflow, exports the best model to ONNX, and serves it from a Rust service with measured CPU latency.

This is an independent portfolio project built on a public benchmark. It is not affiliated with or endorsed by any company.

![PatchCore heatmaps on metal_nut](docs/figures/patchcore_metal_nut.png)

*Top: test images. Bottom: anomaly score in red, true defect outlined in cyan. Examples are picked by a fixed rule that ignores scores, so misses stay visible. Images: MVTec AD, adapted, CC BY-NC-SA 4.0.*

## The problem and why it matters

Manual visual inspection is repetitive and inconsistent, and supervised vision models need many labelled examples of every defect. On a good production line defects are rare, and new kinds keep appearing. Anomaly detection learns what "good" looks like and flags deviations, so:

- **No defect samples are needed to start.** A product can be set up from images of good parts.
- **Previously unseen defects can still be flagged**, because the model does not depend on a list of known types.
- **Heatmaps show where the problem is**, which supports human review and root-cause work.
- **Scrap and rework costs can fall** if defects are caught early, and inspectors can focus on borderline cases.

Metal and pump-like parts in steel and pump manufacturing, for example at companies such as thyssenkrupp or Wilo, are the kind of target industry this approach is relevant to. No company data was used here, and this repository makes no claim about savings at any real plant.

## Architecture

```mermaid
flowchart LR
    subgraph Training["Training (Python, GPU)"]
        D[("MVTec AD<br/>good images")] --> P["Shared preprocessing<br/>resize 256, crop 224"]
        P --> B["Baseline<br/>PaDiM-style Gaussians"]
        P --> C["PatchCore (anomalib)<br/>coreset memory bank"]
        B --> E["Shared evaluation<br/>AUROC, AUPRO, F1"]
        C --> E
        E --> M[("MLflow<br/>metrics, heatmaps, models")]
        C --> X["ONNX export<br/>includes nearest-neighbour search"]
        X --> T{{"Parity check<br/>PyTorch vs ONNX Runtime"}}
        X --> V[("DVC<br/>local remote")]
    end
    subgraph Serving["Serving (Rust, CPU)"]
        X --> S["axum + ort<br/>Pillow-exact preprocessing"]
        S --> A1["POST /predict"]
        S --> A2["GET /health"]
        S --> A3["GET /metrics"]
    end
    Client(["Inspection station"]) --> A1
    A3 --> Prom(["Prometheus"])
```

## Results

All tables between `generated` markers are written by `make report` from MLflow and benchmark files. Per-category details, provenance and all heatmaps are in [docs/results.md](docs/results.md).

<!-- generated:summary:start -->
<!-- generated:summary:end -->

"Focus 6" means metal_nut, screw, pill, transistor, bottle and carpet, chosen as the closest to metal and pump-like parts.

**How to read this:** AUROC measures ranking quality and doesn't require choosing a threshold. A real line needs a yes/no decision, so the calibrated rows are the practical ones. Their threshold comes only from held-out good images, never from test labels. The gap between the two F1 rows shows how much a model depends on a threshold it couldn't choose in production.

Focus categories:

<!-- generated:focus:start -->
<!-- generated:focus:end -->

### Why these models

- **Baseline: PaDiM-style Mahalanobis distance on ResNet18 features.** It fits one Gaussian per patch location in closed form, with no training loop or hyperparameter tuning. It was chosen over a convolutional autoencoder because it is deterministic, fast to fit, and a recognised reference point. A weak autoencoder would make PatchCore look better for the wrong reason.
- **PatchCore: anomalib's model with a WideResNet50 backbone** and a 10% coreset, with anomalib's defaults otherwise. It compares each patch with a memory bank of good patches, so it needs no training but carries that memory bank into inference.

Both use the same loader, preprocessing, anomaly map post-processing and metric code. Hyperparameters were not tuned on the test set. Training is reproducible: `scripts/check_determinism.py` refits a category and confirms that the fitted tensors are identical.

## Latency

<!-- generated:latency:start -->
<!-- generated:latency:end -->

ONNX Runtime threads were pinned with `taskset`: one hardware thread per performance core up to 4 threads, and 6 performance plus 2 efficiency cores at 8 threads, so 8-thread results reflect this hybrid CPU. Other services on the machine were running, and frequency scaling was on. HTTP latency includes PNG decoding, preprocessing and JSON encoding. CPU latency grows with memory bank size, which is why screw is slower than metal_nut. Treat this as one machine's measurement, not a guarantee.

## ONNX export and parity

PatchCore's nearest-neighbour search is part of the exported graph, written as matrix multiplication over the memory bank, so one file holds the whole model. A custom wrapper replaces anomalib's export path, because batching and chunking in the original forward pass do not trace to a single-image graph. Parity is enforced by `tests/test_export.py` and by the export script itself:

<!-- generated:parity:start -->
<!-- generated:parity:end -->

The Rust preprocessing ports Pillow's fixed-point resize, because the `image` crate's resize rounds differently. `tests/test_serving.py` checks that the resulting tensors are **bit-identical** to Python's on images from all 15 categories. This is verified for PNG. JPEG decoders can differ between libraries, so JPEG uploads are not covered by that guarantee.

## Limitations

- **MVTec AD is a benchmark, not a production line.** Images are clean, centred and evenly lit. Results here don't predict performance at a real station.
- **Domain shift.** New lighting, camera position, material batches or wear can raise false alarms. Models need refitting and monitoring when conditions change.
- **Threshold calibration is fragile.** It uses very few images in small categories (6 for toothbrush), and a defect can land just under the threshold. Good images are also held out from fitting, so numbers can differ slightly from published PatchCore results.
- **The centre crop discards the image border.** A 256 to 224 crop means defects at the edge can't be detected and aren't counted either. A real station would need to cover the full part.
- **Single seed, per-category models.** Numbers come from one seed. Each product needs its own model and memory bank, and the Latency section shows the measured memory per loaded model.
- **Throughput.** Each model handles one request at a time, and throughput stays limited by the CPU at this memory bank size.

## From here to a real plant

1. **Camera and lighting:** fixed mounting, controlled light, and resolution matched to the smallest relevant defect, tiling high-resolution images if needed.
2. **Data collection:** good parts across shifts, batches and tool wear, reviewed by quality staff.
3. **Threshold per product:** calibrate on a large good set, validate with real defects, and set the target from the cost of a missed defect versus a false reject.
4. **Human-in-the-loop review:** route borderline parts to inspectors with the heatmap, and feed the decisions back.
5. **Drift monitoring:** alert on shifts in score distribution and false reject rate using the Prometheus metrics, with a defined refit process.
6. **Integration:** connect to line control and MES systems, version models per product, and support rollback.
7. **Speed:** evaluate smaller coresets, an approximate nearest-neighbour index, or a small GPU against the line's cycle time.

## How to reproduce

Requirements: Linux, an NVIDIA GPU with CUDA 12.8+ support, [uv](https://docs.astral.sh/uv/), Docker with Compose, and Rust.

1. Download MVTec AD from <https://www.mvtec.com/company/research/datasets/mvtec-ad> after accepting its license, and extract it so that `data/mvtec_ad/bottle/train/good/000.png` exists.
2. Run the pipeline:

```bash
make setup && make data-check && make train && make eval && make export && make bench && make report
```

`make train` and `make eval` start MLflow at <http://127.0.0.1:5000>. To serve the models and try the service:

```bash
make serve   # 127.0.0.1:8090, change it with INFERENCE_PORT
curl -F image=@data/mvtec_ad/metal_nut/test/scratch/000.png \
  'http://127.0.0.1:8090/predict?category=metal_nut&heatmap=true'
```

`make test` and `make lint` run the checks. Exported models and the dataset pointer are versioned with DVC, on a local remote only.

### Layout

```text
src/anomaly_inspect/   data, models, metrics, pipeline, ONNX export, report generation
configs/               baseline.yaml, patchcore.yaml
scripts/               check_data, train, evaluate, export_onnx, bench, generate_report
serving/               Rust service, multi-stage Dockerfile
tests/                 unit tests and cross-language parity tests
results/, docs/        benchmark output, generated results and figures
```

## Data and license

This project uses the **MVTec Anomaly Detection dataset (MVTec AD)**, licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). That license permits **non-commercial use only**. The dataset is not part of this repository and must be downloaded from MVTec. The heatmap figures in `docs/figures` contain adapted excerpts of the dataset and are shared under the same license.

Please cite:

> Paul Bergmann, Michael Fauser, David Sattlegger, Carsten Steger. *MVTec AD: A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection.* IEEE/CVF CVPR, 2019.

> Paul Bergmann, Kilian Batzner, Michael Fauser, David Sattlegger, Carsten Steger. *The MVTec Anomaly Detection Dataset: A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection.* International Journal of Computer Vision, 2021.

Methods: Defard et al., *PaDiM*, ICPR Workshops 2021. Roth et al., *Towards Total Recall in Industrial Anomaly Detection*, CVPR 2022. PatchCore implementation: [anomalib](https://github.com/open-edge-platform/anomalib).
