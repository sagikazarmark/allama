use allama::decision::Decision;

// One error for the field type, none for the outcome description.
#[derive(Decision)]
struct Triage {
    #[decision(instructions = "Summarize", yes = "Yes")]
    summary: String,
}

fn main() {}
