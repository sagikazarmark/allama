use allama::decision::{Decision, YesNoAnswer};

#[derive(Decision)]
struct BlankOutcome {
    #[decision(instructions = "Spam?", yes = " ")]
    spam: YesNoAnswer,
}

fn main() {}
