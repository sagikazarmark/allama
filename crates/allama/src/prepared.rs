//! What the `Prepared` request of every operation shares: the warnings it
//! collected and when they are emitted as warning events.

use crate::{Degradations, Subject, Warning, warning};

/// The degradations of a prepared request, and the rule for emitting them
/// when the request is dropped.
///
/// A `Prepared` request hands its degradations on when it finishes
/// (`finish`, `finish_stream`, `into_validation`). Dropped before that, it
/// emits them only when the request reached the transport
/// ([`mark_sent`](Self::mark_sent)) and it is not a dry run: the
/// degradations happened even though the call failed afterwards, and a
/// request refused before sending describes a call that never happened.
#[derive(Debug)]
pub(crate) struct PendingWarnings {
    model: String,
    /// `None` only once a consuming method took it.
    degradations: Option<Degradations>,
    dry_run: bool,
    sent: bool,
}

impl PendingWarnings {
    pub(crate) fn new(model: String, degradations: Degradations, dry_run: bool) -> Self {
        Self {
            model,
            degradations: Some(degradations),
            dry_run,
            sent: false,
        }
    }

    pub(crate) fn is_dry_run(&self) -> bool {
        self.dry_run
    }

    pub(crate) fn mark_sent(&mut self) {
        self.sent = true;
    }

    pub(crate) fn degradations_mut(&mut self) -> &mut Degradations {
        self.degradations
            .as_mut()
            .expect("only a consuming method takes the degradations")
    }

    /// Hand the degradations on, with what is needed to emit them.
    pub(crate) fn take(mut self) -> Taken {
        Taken {
            degradations: self
                .degradations
                .take()
                .expect("only a consuming method takes the degradations"),
            model: std::mem::take(&mut self.model),
            emit: !self.dry_run,
        }
    }
}

impl Drop for PendingWarnings {
    fn drop(&mut self) {
        if self.dry_run || !self.sent {
            return;
        }
        if let Some(degradations) = self.degradations.take() {
            let provider = degradations.provider().to_owned();
            for mut warning in degradations.into_warnings() {
                warning::emit(&provider, &self.model, &mut warning);
            }
        }
    }
}

/// Degradations a `Prepared` request handed on.
pub(crate) struct Taken {
    pub(crate) degradations: Degradations,
    pub(crate) model: String,
    /// Whether its warnings are emitted (not for a dry run).
    pub(crate) emit: bool,
}

impl Taken {
    /// The request-side warnings, then `inner` translated to the caller's
    /// indices, each emitted unless this is a dry run.
    pub(crate) fn merge(self, inner: Vec<Warning>) -> Vec<Warning> {
        let provider = self.degradations.provider().to_owned();
        let translated: Vec<Warning> = inner
            .into_iter()
            .map(|warning| self.degradations.translate(warning))
            .collect();
        let mut warnings = self.degradations.into_warnings();
        warnings.extend(translated);
        if self.emit {
            for warning in &mut warnings {
                warning::emit(&provider, &self.model, warning);
            }
        }
        warnings
    }

    /// This layer's dry run, followed by `inner`, the dry run of the
    /// provider it forwards to, with `inner`'s warnings translated to the
    /// caller's indices. Nothing is emitted.
    pub(crate) fn merge_validation<C: crate::Capability>(
        self,
        inner: crate::Validation<C>,
    ) -> crate::Validation<C> {
        let unknown = self.degradations.unknown::<C>();
        let inner = inner.map_warnings(|warning| self.degradations.translate(warning));
        crate::Validation::from_parts(
            self.degradations.into_warnings(),
            unknown,
            crate::ValidationCoverage::Capabilities,
        )
        .followed_by(inner)
    }

    /// The recorded `Unknown` capabilities and the warnings, without
    /// emitting anything.
    pub(crate) fn into_validation_parts<C: crate::Capability>(
        self,
    ) -> (Vec<Warning>, Vec<(C, Subject)>) {
        let unknown = self.degradations.unknown::<C>();
        (self.degradations.into_warnings(), unknown)
    }
}
