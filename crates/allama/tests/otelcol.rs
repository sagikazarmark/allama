//! The shipped OpenTelemetry Collector config
//! (`docs/guides/observability/otelcol-genai-metrics.yaml`), run by
//! `otelcol-contrib`.
//!
//! Ignored by default: they need an `otelcol-contrib` binary, on `PATH` or
//! named by `OTELCOL_CONTRIB`. The CI "Collector" job installs the pinned
//! version and runs them with `cargo test -p allama --test otelcol --
//! --ignored`.

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use serde_json::Value;

#[test]
#[ignore = "needs otelcol-contrib; run by the CI Collector job"]
fn shipped_config_validates() {
    let status = Command::new(otelcol())
        .arg("validate")
        .arg("--config")
        .arg(shipped_config())
        .status()
        .expect("run otelcol-contrib");
    assert!(
        status.success(),
        "otelcol-contrib validate failed: {status}"
    );
}

/// A span `FallbackProvider` served from its fallback yields metrics
/// attributed to the serving provider (`allama.response.provider`), not the
/// addressed one; a chain where every side failed is attributed to the side
/// whose error ended it; and a cancelled call keeps its
/// `allama.response.cancelled` dimension.
#[test]
#[ignore = "needs otelcol-contrib; run by the CI Collector job"]
fn metrics_attribute_a_failover_to_the_serving_provider() {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("otelcol");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    // The JSON file receiver reads one OTLP request per line.
    let fixture: Value =
        serde_json::from_str(&std::fs::read_to_string(fixture("failover-spans.json")).unwrap())
            .unwrap();
    let spans = dir.join("spans.jsonl");
    std::fs::write(&spans, format!("{fixture}\n")).unwrap();

    // Read the spans from the file instead of OTLP, and write the metrics
    // to a file instead of the debug exporter; the processors and the
    // connector are the shipped ones.
    let metrics = dir.join("metrics.jsonl");
    let overrides = dir.join("overrides.yaml");
    std::fs::write(
        &overrides,
        format!(
            r#"receivers:
  otlp_json_file:
    include: ["{spans}"]
    start_at: beginning
exporters:
  file:
    path: "{metrics}"
service:
  telemetry:
    metrics:
      level: none
  pipelines:
    traces:
      receivers: [otlp_json_file]
    metrics:
      exporters: [file]
"#,
            spans = spans.display(),
            metrics = metrics.display(),
        ),
    )
    .unwrap();

    let mut collector = Command::new(otelcol())
        .arg("--config")
        .arg(shipped_config())
        .arg("--config")
        .arg(&overrides)
        .stdout(Stdio::null())
        .stderr(Stdio::inherit())
        .spawn()
        .expect("run otelcol-contrib");
    let points = wait_for_duration_points(&metrics, Duration::from_secs(60));
    let _ = collector.kill();
    let _ = collector.wait();

    let mut providers: Vec<_> = points
        .iter()
        .map(|point| attribute(point, "gen_ai.provider.name").unwrap_or_default())
        .collect();
    providers.sort();
    providers.dedup();
    assert_eq!(providers, ["gcp.gemini", "openai"], "{points:#?}");

    let cancelled: Vec<_> = points
        .iter()
        .filter(|point| attribute(point, "allama.response.cancelled").is_some())
        .map(|point| attribute(point, "gen_ai.provider.name"))
        .collect();
    assert_eq!(cancelled, [Some("gcp.gemini".to_owned())], "{points:#?}");

    // Both sides failed: the fallback's auth error, not the addressed
    // primary.
    let failed: Vec<_> = points
        .iter()
        .filter(|point| attribute(point, "error.type").is_some())
        .map(|point| {
            (
                attribute(point, "gen_ai.provider.name"),
                attribute(point, "error.type"),
            )
        })
        .collect();
    assert_eq!(
        failed,
        [(Some("openai".to_owned()), Some("auth".to_owned()))],
        "{points:#?}"
    );
}

/// The binary to run.
fn otelcol() -> String {
    std::env::var("OTELCOL_CONTRIB").unwrap_or_else(|_| "otelcol-contrib".to_owned())
}

fn shipped_config() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../docs/guides/observability/otelcol-genai-metrics.yaml")
}

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/otlp")
        .join(name)
}

/// The data points of `gen_ai.client.operation.duration` the collector
/// wrote to `path`, once every fixture span is in.
fn wait_for_duration_points(path: &Path, timeout: Duration) -> Vec<Value> {
    let started = Instant::now();
    loop {
        let points = duration_points(path);
        if points.len() >= 3 {
            return points;
        }
        assert!(
            started.elapsed() < timeout,
            "no metrics after {timeout:?}; got {points:#?}"
        );
        std::thread::sleep(Duration::from_millis(200));
    }
}

fn duration_points(path: &Path) -> Vec<Value> {
    let Ok(contents) = std::fs::read_to_string(path) else {
        return Vec::new();
    };
    let mut points = Vec::new();
    for line in contents.lines() {
        // The last line may still be being written.
        let Ok(request) = serde_json::from_str::<Value>(line) else {
            continue;
        };
        for metric in array(&request, "resourceMetrics")
            .iter()
            .flat_map(|resource| array(resource, "scopeMetrics"))
            .flat_map(|scope| array(scope, "metrics"))
            .filter(|metric| metric["name"] == "gen_ai.client.operation.duration")
        {
            points.extend(array(&metric["histogram"], "dataPoints").iter().cloned());
        }
    }
    points
}

fn array<'a>(value: &'a Value, key: &str) -> &'a [Value] {
    value[key].as_array().map_or(&[], Vec::as_slice)
}

/// A data point attribute, as a string.
fn attribute(point: &Value, key: &str) -> Option<String> {
    array(point, "attributes")
        .iter()
        .find(|attribute| attribute["key"] == key)
        .map(|attribute| match &attribute["value"] {
            Value::Object(value) => value
                .values()
                .next()
                .map(|value| match value {
                    Value::String(text) => text.clone(),
                    other => other.to_string(),
                })
                .unwrap_or_default(),
            other => other.to_string(),
        })
}
