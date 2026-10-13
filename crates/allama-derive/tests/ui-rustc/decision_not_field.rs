use allama::decision::{Choice, Decision};

#[derive(Decision)]
struct Triage {
    #[decision(instructions = "Summarize")]
    summary: String,
    #[decision(instructions = "Pick one")]
    pick: Choice<bool>,
}

fn main() {}
