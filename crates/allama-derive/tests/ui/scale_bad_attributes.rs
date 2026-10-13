use allama::decision::Scale;

#[derive(Scale)]
#[decision(rename_all = "lowercase")]
enum Urgency {
    Low,
    High,
}

#[derive(Scale)]
enum Blank {
    #[decision(description = " ")]
    Low,
    High,
}

fn main() {}
