use allama::decision::ChoiceSet;

#[derive(ChoiceSet)]
#[decision(rename_all = "snake_case")]
enum Weird {
    __,
    A,
}

#[derive(ChoiceSet)]
enum Blank {
    #[decision(rename = " ")]
    A,
    #[decision(description = "  ")]
    B,
}

fn main() {}
