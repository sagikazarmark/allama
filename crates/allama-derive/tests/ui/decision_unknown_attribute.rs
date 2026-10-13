use allama::decision::{Decision, YesNoAnswer};

#[derive(Decision)]
struct Triage {
    #[decision(instructions = "Refund?", maybe = "Unsure")]
    refund: YesNoAnswer,
}

#[derive(Decision)]
#[decision(rename_all = "lowercase")]
struct Container {
    #[decision(instructions = "Refund?")]
    refund: YesNoAnswer,
}

fn main() {}
