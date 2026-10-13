use allama::decision::ChoiceSet;

#[derive(ChoiceSet)]
#[decision(rename_all = "Title Case")]
enum RenameAll {
    A,
    B,
}

#[derive(ChoiceSet)]
enum Unknown {
    #[decision(describe = "typo")]
    A,
    B,
}

#[derive(ChoiceSet)]
enum Duplicate {
    #[decision(description = "a", description = "b")]
    A,
    B,
}

fn main() {}
