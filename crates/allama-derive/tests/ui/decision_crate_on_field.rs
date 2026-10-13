use allama::decision::{Decision, YesNoAnswer};

#[derive(Decision)]
struct Triage {
    /// Is this spam?
    #[decision(crate = "allama")]
    spam: YesNoAnswer,
}

fn main() {}
