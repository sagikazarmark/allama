use allama::decision::{Decision, DecisionField};

// The non-generic field is reported once, at the field.
#[derive(Decision)]
struct Triage<T: DecisionField> {
    #[decision(instructions = "Anything?")]
    any: T,
    #[decision(instructions = "Summarize")]
    summary: String,
}

fn main() {}
