use allama::decision::{Decision, YesNoAnswer};

#[derive(Decision)]
struct Triage {
    #[decision(instructions = "Refund?")]
    refund: YesNoAnswer,
    #[decision(rename = "refund", instructions = "Refund again?")]
    refund_again: YesNoAnswer,
}

fn main() {}
