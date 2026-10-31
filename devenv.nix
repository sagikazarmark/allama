{
  config,
  lib,
  pkgs,
  ...
}:

{
  dotenv.enable = true;

  packages =
    with pkgs;
    [
      jq
      just

      cargo-audit
      cargo-deny
      cargo-expand
      cargo-hack
      cargo-release
      cargo-semver-checks
      cargo-watch
    ]
    # The `native-tls` features (on in `--all-features` builds) link the
    # system OpenSSL on Linux.
    ++ lib.optionals stdenv.isLinux [
      openssl
      pkg-config
    ];

  # Builds the example Worker (crates/allama-examples-worker), at the version
  # of `worker` in Cargo.lock.
  worker-build = {
    enable = true;
  };

  # `devenv up` serves the example Worker under `wrangler dev` (process
  # `wrangler-worker`).
  wrangler = {
    enable = true;

    dev.worker.directory = "crates/allama-examples-worker";
  };

  # The shell's RUSTFLAGS make the host linker lld (`-fuse-ld=lld`), an
  # argument rust-lld rejects when it links wasm; the Worker is all wasm.
  processes.wrangler-worker.env.RUSTFLAGS = "";

  # `devenv test` starts the example Worker, then runs its checks against it.
  enterTest = ''
    just worker-smoke "http://127.0.0.1:${toString config.processes.wrangler-worker.ports.http.value}"
  '';

  languages = {
    rust = {
      enable = true;
      channel = "stable";
      # Pinned so that a new stable release (new lints, new rustdoc checks)
      # lands in its own pull request; bump it on purpose.
      version = "1.98.1";
      targets = [
        "wasm32-unknown-unknown"
        "x86_64-unknown-linux-musl"
      ];

      lld.enable = true;
    };
  };

  services = {
    ollama = {
      enable = true;
    };

    litellm = {
      enable = true;
    };
  };

  profiles = {
    # CI runs no services: leave their packages out of the shell.
    ci.module = {
      services.ollama.enable = lib.mkForce false;
      services.litellm.enable = lib.mkForce false;
    };

    # The minimum supported Rust version, in place of the pinned toolchain.
    msrv = {
      extends = [ "ci" ];
      module = {
        languages.rust.version = lib.mkForce (
          lib.versions.pad 3 (fromTOML (builtins.readFile ./Cargo.toml)).workspace.package.rust-version
        );
      };
    };
  };
}
