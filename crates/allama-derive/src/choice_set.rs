use proc_macro2::{Literal, TokenStream};
use quote::quote;
use syn::DeriveInput;

use crate::attr;

pub fn expand(input: DeriveInput) -> syn::Result<TokenStream> {
    let (entries, krate) = attr::entries(&input, "ChoiceSet", "option key")?;

    let options = entries.iter().map(|entry| {
        let (key, description) = (&entry.name, &entry.description);
        quote!((#key, #description))
    });
    let name = &input.ident;
    let idents: Vec<_> = entries.iter().map(|entry| &entry.variant.ident).collect();
    let indices: Vec<_> = (0..entries.len()).map(Literal::usize_unsuffixed).collect();

    Ok(quote! {
        impl #krate::decision::ChoiceSet for #name {
            const OPTIONS: &'static [(&'static str, ::core::option::Option<&'static str>)] =
                &[#(#options),*];

            fn from_index(index: usize) -> ::core::option::Option<Self> {
                match index {
                    #(#indices => ::core::option::Option::Some(Self::#idents),)*
                    _ => ::core::option::Option::None,
                }
            }

            fn index(&self) -> usize {
                match self {
                    #(Self::#idents => #indices,)*
                }
            }
        }

        impl #krate::decision::DecisionField for #name {
            const KIND: #krate::decision::QuestionKind = #krate::decision::QuestionKind::Choice;

            fn question(
                instructions: impl ::core::convert::Into<::std::string::String>,
            ) -> #krate::decision::Question {
                <#krate::decision::Choice<Self> as #krate::decision::DecisionField>::question(
                    instructions,
                )
            }

            fn from_answer(
                answer: &#krate::decision::Answer,
            ) -> ::core::result::Result<Self, #krate::decision::AnswerMismatch> {
                <#krate::decision::Choice<Self> as #krate::decision::DecisionField>::from_answer(
                    answer,
                )
                .map(#krate::decision::Choice::into_value)
            }
        }
    })
}
