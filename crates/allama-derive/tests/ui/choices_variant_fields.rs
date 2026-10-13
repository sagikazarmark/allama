use allama::decision::ChoiceSet;

#[derive(ChoiceSet)]
enum Label {
    Billing,
    Other(String),
}

fn main() {}
