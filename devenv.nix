{ lib, pkgs, ... }:

{
  dotenv.enable = true;

  packages = with pkgs; [
    jq
    just

    cargo-audit
    cargo-deny
    cargo-expand
    cargo-hack
    cargo-release
    cargo-semver-checks
    cargo-watch
  ];

  worker-build = {
    enable = true;
  };

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
        languages.rust.version =
          lib.mkForce lib.versions.pad 3
            (fromTOML (builtins.readFile ./Cargo.toml)).workspace.package.rust-version;
      };
    };
  };
}
