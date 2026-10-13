use allama::decision::{Decision, YesNoAnswer};

#[derive(Decision)]
struct Triage {
    refund: YesNoAnswer,
}

#[derive(Decision)]
struct Blank {
    #[decision(instructions = "  ")]
    refund: YesNoAnswer,
}

#[derive(Decision)]
struct BlankDoc {
    ///
    refund: YesNoAnswer,
}

fn main() {}
