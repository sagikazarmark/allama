use proc_macro2::{Literal, TokenStream};
use quote::quote;
use syn::DeriveInput;

use crate::attr;

pub fn expand(input: DeriveInput) -> syn::Result<TokenStream> {
    let (entries, krate) = attr::entries(&input, "Scale", "level label")?;

    let levels = entries.iter().map(|entry| {
        let (label, description) = (&entry.name, &entry.description);
        quote!((#label, #description))
    });
    let name = &input.ident;
    let idents: Vec<_> = entries.iter().map(|entry| &entry.variant.ident).collect();
    let indices: Vec<_> = (0..entries.len()).map(Literal::usize_unsuffixed).collect();

    Ok(quote! {
        impl #krate::decision::Scale for #name {
            const LEVELS: &'static [(&'static str, ::core::option::Option<&'static str>)] =
                &[#(#levels),*];

            fn from_level(level: usize) -> ::core::option::Option<Self> {
                match level {
                    #(#indices => ::core::option::Option::Some(Self::#idents),)*
                    _ => ::core::option::Option::None,
                }
            }

            fn level(&self) -> usize {
                match self {
                    #(Self::#idents => #indices,)*
                }
            }
        }
    })
}
