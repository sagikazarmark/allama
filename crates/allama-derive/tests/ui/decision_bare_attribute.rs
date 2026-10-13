use allama::decision::{Decision, YesNoAnswer};

#[derive(Decision)]
struct Triage {
    /// Is this spam?
    #[decision]
    spam: YesNoAnswer,
}

fn main() {}
