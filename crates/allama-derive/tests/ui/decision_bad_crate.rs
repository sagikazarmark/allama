use allama::decision::{Decision, YesNoAnswer};

#[derive(Decision)]
#[decision(crate = "not a path")]
struct Triage {
    #[decision(instructions = "Spam?")]
    spam: YesNoAnswer,
}

fn main() {}
