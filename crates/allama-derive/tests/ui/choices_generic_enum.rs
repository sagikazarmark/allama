use allama::decision::ChoiceSet;

#[derive(ChoiceSet)]
enum Label<T> {
    Bug,
    Feature,
    #[allow(dead_code)]
    Other(std::marker::PhantomData<T>),
}

fn main() {}
