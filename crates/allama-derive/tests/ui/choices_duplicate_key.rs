use allama::decision::ChoiceSet;

#[derive(ChoiceSet)]
#[decision(rename_all = "lowercase")]
enum Label {
    Billing,
    #[decision(rename = "billing")]
    Payments,
}

fn main() {}
