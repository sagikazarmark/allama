use allama::decision::{Decision, YesNoAnswer};

#[derive(Decision)]
struct Triage {
    #[decision(instructions = 42)]
    spam: YesNoAnswer,
}

fn main() {}
