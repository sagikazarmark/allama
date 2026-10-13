use allama::decision::{ChoiceSet, Decision};

#[derive(ChoiceSet)]
enum Label {
    Billing,
    Bug,
}

#[derive(Decision)]
struct Triage {
    #[decision(instructions = "Which label?", yes = "Billing")]
    label: Label,
}

fn main() {
    let _ = Triage::questions();
}
