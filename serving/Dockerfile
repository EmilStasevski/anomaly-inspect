# Build the release binary, then copy it into a minimal runtime image.
# The prebuilt ONNX Runtime linked by `ort` needs glibc 2.38+, so both stages use Debian 13.
FROM rust:1.99-trixie AS builder
WORKDIR /build
COPY Cargo.toml Cargo.lock ./
COPY src ./src
RUN cargo build --release --locked --bin anomaly-serving

# Contains glibc and libstdc++, which ONNX Runtime needs, and no shell. Runs as UID 65532.
FROM gcr.io/distroless/cc-debian13:nonroot
COPY --from=builder /build/target/release/anomaly-serving /usr/local/bin/anomaly-serving
ENV MODEL_ROOT=/models BIND_ADDRESS=0.0.0.0:8080
EXPOSE 8080
HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=3 \
    CMD ["/usr/local/bin/anomaly-serving", "healthcheck"]
ENTRYPOINT ["/usr/local/bin/anomaly-serving"]
