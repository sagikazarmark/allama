//! Every wrapper is `Clone` when what it wraps is, so a stack can be built
//! once and cloned per task.

use allama::wrappers::{
    CapabilityOverrideProvider, ChatExchange, FallbackProvider, MapModelProvider,
    MapRequestProvider, ObserveProvider, RecordingProvider, RetryingProvider, TimeoutProvider,
    UnsupportedPolicyProvider,
};

fn assert_clone<T: Clone>() {}

fn assert_clone_value<T: Clone>(_: &T) {}

#[test]
fn wrappers_over_clone_providers_are_clone() {
    assert_clone::<TimeoutProvider<()>>();
    assert_clone::<FallbackProvider<(), ()>>();
    assert_clone::<MapModelProvider<(), fn(&str) -> String>>();
    assert_clone::<MapRequestProvider<(), fn(&mut allama::ChatRequest)>>();
    assert_clone::<ObserveProvider<()>>();
    assert_clone::<RetryingProvider<()>>();
    assert_clone::<CapabilityOverrideProvider<()>>();
    assert_clone::<UnsupportedPolicyProvider<()>>();
    #[cfg(feature = "tracing")]
    assert_clone::<allama::wrappers::TracingProvider<()>>();
}

/// A recorder is `Clone` even when its sink is not: clones share the sink.
#[test]
fn a_recorder_with_a_sink_that_is_not_clone_is_clone() {
    let sink = std::sync::Mutex::new(Vec::<ChatExchange>::new());
    let recorder = RecordingProvider::new((), move |exchange| sink.lock().unwrap().push(exchange));
    assert_clone_value(&recorder);
}
