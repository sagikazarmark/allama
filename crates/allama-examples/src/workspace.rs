//! The crates of the workspace, for the tests that read every crate
//! (`tests/warning_codes.rs`, `tests/guide_anchors.rs`). They come from
//! `cargo metadata`, so the tests hold no opinion on where the crates live.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

use serde_json::Value;

/// The workspace this crate is built in.
#[derive(Debug)]
pub struct Workspace {
    /// The repository root: the directory of the workspace `Cargo.toml`.
    pub root: PathBuf,
    /// Every crate in the workspace.
    pub packages: Vec<Package>,
}

/// A crate in the workspace.
#[derive(Debug)]
pub struct Package {
    /// The package name, as on crates.io.
    pub name: String,
    /// The directory holding the package's `Cargo.toml`.
    pub root: PathBuf,
    /// Whether the package is published; `false` under `publish = false`.
    pub published: bool,
}

impl Workspace {
    /// The workspace, read once from `cargo metadata`.
    ///
    /// # Panics
    ///
    /// When `cargo metadata` fails: this crate is never built outside the
    /// workspace.
    #[must_use]
    pub fn get() -> &'static Self {
        static WORKSPACE: OnceLock<Workspace> = OnceLock::new();
        WORKSPACE.get_or_init(Self::load)
    }

    /// The crate named `name`.
    ///
    /// # Panics
    ///
    /// When the workspace has no such crate.
    #[must_use]
    pub fn package(&self, name: &str) -> &Package {
        self.packages
            .iter()
            .find(|package| package.name == name)
            .unwrap_or_else(|| panic!("the workspace has a crate named `{name}`"))
    }

    fn load() -> Self {
        let output = Command::new(env!("CARGO"))
            .args([
                "metadata",
                "--no-deps",
                "--offline",
                "--format-version",
                "1",
            ])
            .args([
                "--manifest-path",
                concat!(env!("CARGO_MANIFEST_DIR"), "/Cargo.toml"),
            ])
            .output()
            .expect("cargo runs");
        assert!(
            output.status.success(),
            "cargo metadata failed:\n{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let metadata: Value =
            serde_json::from_slice(&output.stdout).expect("cargo metadata is JSON");
        let packages = metadata["packages"]
            .as_array()
            .expect("the metadata lists packages")
            .iter()
            .map(|package| {
                let manifest = package["manifest_path"]
                    .as_str()
                    .expect("a package has a manifest path");
                Package {
                    name: package["name"]
                        .as_str()
                        .expect("a package has a name")
                        .to_owned(),
                    root: Path::new(manifest)
                        .parent()
                        .expect("a manifest is in a directory")
                        .to_path_buf(),
                    // `publish` is null when unrestricted, and the allowed
                    // registries otherwise: none under `publish = false`.
                    published: package["publish"]
                        .as_array()
                        .is_none_or(|registries| !registries.is_empty()),
                }
            })
            .collect();
        Self {
            root: PathBuf::from(
                metadata["workspace_root"]
                    .as_str()
                    .expect("the metadata names the workspace root"),
            ),
            packages,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_workspace_lists_this_crate_and_the_published_ones() {
        let workspace = Workspace::get();
        assert!(workspace.root.join("Cargo.toml").is_file());

        let this = workspace.package("allama-examples");
        assert_eq!(this.root, Path::new(env!("CARGO_MANIFEST_DIR")));
        assert!(!this.published);

        let core = workspace.package("allama");
        assert!(core.published);
        assert!(core.root.join("src/lib.rs").is_file());
        assert!(core.root.starts_with(&workspace.root));
    }
}
