# List the recipes.
[private]
default:
    @just --list --unsorted

# musl, collector, msrv and semver-checks need tools, a toolchain or a baseline
# beyond the default shell; see their recipes.
# Run the CI jobs that need no extra setup.
ci: build test examples benchmarks docs features provider-features footprint package wasm fmt clippy unused-dependencies audit deny

# Build the workspace.
build:
    cargo build --workspace --all-targets --all-features --locked

# Doc tests and integration tests gate on features (`mock`,
# `warning-events`, ...): with the default set, the gates themselves are what
# is tested.
# Run tests.
test:
    cargo test --workspace --all-features --locked
    cargo test -p allama --locked

# Most allama-examples examples require live provider credentials, so only the
# allow-listed credential-free ones run here. The Examples workflow runs the
# rest per provider with secrets.
# Run the examples that need no credentials.
examples:
    #!/usr/bin/env bash
    set -euo pipefail
    cargo metadata --no-deps --format-version 1 --locked \
      | jq -r '["middleware", "text_gateway", "tower_transport", "trace_context"] as $credential_free | .packages[] as $package | $package.targets[] | select(.kind[] == "example") | select($package.name != "allama-examples" or (.name | IN($credential_free[]))) | [$package.name, .name] | @tsv' \
      | while IFS=$'\t' read -r package example; do
          cargo run -p "$package" --all-features --example "$example" --locked
        done

# Run benchmarks.
benchmarks:
    cargo bench --workspace --all-features --locked

# docs.rs builds with all features, but users read `cargo doc` output with
# defaults: intra-doc links to feature-gated items must not break. cargo hack
# documents each published crate on its own, so feature unification across
# the workspace cannot turn on the feature a link's target needs.
# Build the documentation.
[env("RUSTDOCFLAGS", "--deny warnings")]
docs:
    cargo doc --workspace --all-features --no-deps --locked
    cargo doc --workspace --no-deps --locked
    cargo hack doc --workspace --ignore-private --no-default-features --no-deps --locked

# Code left unused without the default features is a gating bug. The allama
# feature powerset, two features at a time, catches leaks where a non-gated
# symbol references a type that only exists under some but not all of the
# features (e.g. impls that need two features simultaneously).
# Check feature combinations.
features:
    cargo check --workspace --all-targets --locked
    RUSTFLAGS="--deny warnings" cargo check --workspace --all-targets --no-default-features --locked
    cargo hack check -p allama --feature-powerset --depth 2 --exclude-no-default-features --exclude-all-features --all-targets --locked

# Each feature on its own, without the defaults, so a feature that only
# compiles next to another one is caught.
# Check each feature of the HTTP, provider and conformance crates.
[env("RUSTFLAGS", "--deny warnings")]
provider-features:
    cargo hack check -p allama-http -p allama-anthropic -p allama-cloudflare -p allama-gemini -p allama-openai -p allama-openai-compat -p allama-systemone -p allama-providers -p allama-conformance --each-feature --all-targets --locked

# With default features off, a provider crate must not depend on reqwest, and
# must work over an injected transport (the tests use ReplayTransport; the
# ones that need reqwest are gated on it). allama-http's `tower` feature is
# off by default and must add no dependency then.
# Check the provider crates without reqwest, and allama-http without tower.
footprint:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "==> allama-http"
    if cargo tree -p allama-http --no-default-features -e normal --target all --locked --prefix none | grep -E "^(http-body|http-body-util|tower-service) "; then
      echo "allama-http depends on the tower feature's crates with it off"
      exit 1
    fi
    if cargo tree -p allama-http -e normal --depth 1 --target all --locked --prefix none | grep -E "^(http-body|http-body-util|tower-service) "; then
      echo "allama-http depends on the tower feature's crates directly with it off"
      exit 1
    fi
    for crate in allama-anthropic allama-cloudflare allama-gemini allama-openai allama-openai-compat allama-systemone; do
      echo "==> $crate"
      if cargo tree -p "$crate" --no-default-features -e normal --locked | grep reqwest; then
        echo "$crate depends on reqwest with default features off"
        exit 1
      fi
      cargo test -p "$crate" --no-default-features --lib --tests --locked
    done

# Needs musl-gcc (musl-tools) and a Linux host. Runners have cmake installed,
# so check the graph too: `rustls-ring` must not pull in aws-lc-sys or anything
# that needs cmake.
# Build the providers for musl with ring TLS.
[env("CC_x86_64_unknown_linux_musl", "musl-gcc")]
[env("CARGO_TARGET_X86_64_UNKNOWN_LINUX_MUSL_LINKER", "musl-gcc")]
musl:
    #!/usr/bin/env bash
    set -euo pipefail
    features=anthropic,cloudflare,gemini,openai,openai-compat,systemone,rustls-ring
    target=x86_64-unknown-linux-musl
    if cargo tree -p allama-providers --no-default-features --features "$features" --target "$target" -e normal,build --locked | grep -E "aws-lc-sys|cmake"; then
      echo "the ring TLS build still needs aws-lc-sys or cmake"
      exit 1
    fi
    cargo build -p allama-providers --no-default-features --features "$features" --target "$target" --locked

# Needs otelcol-contrib on the PATH. Runs the shipped config on an OTLP span
# fixture and checks the provider the derived metrics are attributed to.
# Test the shipped collector config.
collector:
    otelcol-contrib validate --config docs/guides/observability/otelcol-genai-metrics.yaml
    cargo test -p allama --test otelcol --locked -- --ignored

# Packages and verify-builds every published crate, resolving the unpublished
# workspace dependencies locally: catches an include_str! that leaves its
# package, a path-only normal dependency, or a file missing from
# `include`/`exclude` before release day. allama-examples is not published
# (`cargo package` doesn't skip it by itself). A fresh target directory:
# Cargo would reuse an earlier build of a packaged crate of the same version,
# whatever its source now.
# Package the workspace.
package:
    CARGO_TARGET_DIR="$(mktemp -d)" cargo package --workspace --exclude allama-examples --locked

# Run in the msrv devenv profile: `devenv --profile msrv shell -- just msrv`.
# Check with the minimum supported Rust version.
msrv:
    cargo check --workspace --all-features --locked

# Every published library crate, one at a time so feature unification cannot
# hide a crate that only builds next to another. allama-derive is a proc macro
# (it runs on the host); allama-conformance is native test tooling (a
# dev-dependency; its default `server` feature runs a local HTTP server). The
# tests run on Node, with the wasm-bindgen-test-runner that matches the
# lockfile, installed under target/. The devenv shell's RUSTFLAGS make the
# host linker lld (`-fuse-ld=lld`), an argument rust-lld rejects when it links
# wasm (wasm-streams is a cdylib), so they are cleared.
# Build and test for wasm32-unknown-unknown.
[env("CARGO_TARGET_WASM32_UNKNOWN_UNKNOWN_RUNNER", "wasm-bindgen-test-runner")]
[env("RUSTFLAGS", "")]
wasm:
    #!/usr/bin/env bash
    set -euo pipefail
    cargo hack build --target wasm32-unknown-unknown --workspace --ignore-private --exclude allama-conformance --exclude allama-derive --locked
    cargo hack clippy --target wasm32-unknown-unknown --workspace --ignore-private --exclude allama-conformance --exclude allama-derive --all-features --locked -- --deny warnings
    version=$(cargo metadata --format-version 1 --locked | jq -r '.packages[] | select(.name == "wasm-bindgen") | .version')
    cargo install wasm-bindgen-cli --version "$version" --root target/tools --locked
    export PATH="$PWD/target/tools/bin:$PATH"
    cargo test --target wasm32-unknown-unknown -p allama-cloudflare-worker --lib --locked
    cargo test --target wasm32-unknown-unknown -p allama-http --lib --locked
    cargo test --target wasm32-unknown-unknown -p allama-http --lib --all-features --locked

# Check formatting.
fmt:
    cargo fmt --all --check

# Run clippy.
clippy:
    cargo clippy --workspace --all-targets --all-features --locked -- --deny warnings

# rustc's own lint, on library targets: a normal dependency only tests use is
# reported (dev-dependencies don't reach the lib). Text-based tools
# (cargo-machete) count a use in `#[cfg(test)]` code as a use; cargo-udeps
# needs nightly. Only workspace members are linted: cargo caps lints for
# registry dependencies. allama-examples is excluded: it is not published, and
# its library only holds doc tests of the guides and READMEs, so its
# dependencies serve the examples and doc tests.
# Check for unused dependencies.
[env("RUSTFLAGS", "--deny unused_crate_dependencies")]
unused-dependencies:
    cargo check --workspace --exclude allama-examples --lib --all-features --locked
    cargo check --workspace --exclude allama-examples --lib --locked
    cargo check --workspace --exclude allama-examples --lib --no-default-features --locked

# Audit dependencies for security advisories.
audit:
    cargo audit

# Check dependencies against deny.toml.
deny:
    cargo deny check

# Compares with the latest version on crates.io, or with a revision:
# `just semver-checks --baseline-rev main`.
# Check the public API for semver violations.
semver-checks *args:
    cargo semver-checks --workspace --all-features {{ args }}
