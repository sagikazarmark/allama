use allama::decision::Scale;

#[derive(Scale)]
enum Urgency<const N: usize> {
    Low,
    High,
}

fn main() {}
