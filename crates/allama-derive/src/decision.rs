use std::collections::HashSet;

use proc_macro2::{Span, TokenStream, TokenTree};
use quote::{quote, quote_spanned};
use syn::ext::IdentExt;
use syn::spanned::Spanned;
use syn::{Data, DeriveInput, Fields, Ident, LitStr, Type};

use crate::attr;

pub fn expand(mut input: DeriveInput) -> syn::Result<TokenStream> {
    let fields = match &input.data {
        Data::Struct(data) => match &data.fields {
            Fields::Named(fields) => fields.named.clone(),
            _ => {
                return Err(syn::Error::new_spanned(
                    &input.ident,
                    "`Decision` can only be derived for structs with named fields",
                ));
            }
        },
        _ => {
            return Err(syn::Error::new_spanned(
                &input.ident,
                "`Decision` can only be derived for structs",
            ));
        }
    };

    if fields.is_empty() {
        return Err(syn::Error::new_spanned(
            &input.ident,
            "`Decision` structs need at least one field",
        ));
    }

    let mut krate = None;
    attr::parse(&input.attrs, |meta| {
        if meta.path.is_ident("crate") {
            attr::set(&mut krate, &meta)
        } else {
            Err(attr::unknown(&meta, "`crate`"))
        }
    })?;
    let krate = attr::crate_path(krate)?;

    // Names that make a field type depend on the struct's generics. A false
    // positive only moves the field's check to the where clause.
    let generics = input.generics.clone();
    let generic_names: HashSet<String> = generics
        .type_params()
        .map(|param| param.ident.to_string())
        .chain(generics.const_params().map(|param| param.ident.to_string()))
        .chain(
            generics
                .lifetimes()
                .map(|param| param.lifetime.ident.to_string()),
        )
        .chain(std::iter::once("Self".to_owned()))
        .collect();
    // One identifier for the parameter and its uses, so it resolves the same
    // whatever hygiene context the field types come from. Items at the call
    // site still resolve there, so the name is one no caller defines: a
    // `const response` would turn a plain `response` parameter into a pattern.
    let response = Ident::new("__allama_decision_response", Span::mixed_site());

    let mut names = HashSet::new();
    let mut assertions = Vec::new();
    let mut generic_fields = Vec::new();
    let mut questions = Vec::new();
    let mut decoders = Vec::new();
    for field in &fields {
        let (mut instructions, mut rename, mut yes, mut no) = (None, None, None, None);
        attr::parse(&field.attrs, |meta| {
            if meta.path.is_ident("instructions") {
                attr::set(&mut instructions, &meta)
            } else if meta.path.is_ident("rename") {
                attr::set(&mut rename, &meta)
            } else if meta.path.is_ident("yes") {
                attr::set(&mut yes, &meta)
            } else if meta.path.is_ident("no") {
                attr::set(&mut no, &meta)
            } else {
                Err(attr::unknown(
                    &meta,
                    "`instructions`, `rename`, `yes`, or `no`",
                ))
            }
        })?;

        let ident = field.ident.as_ref().expect("named field");
        let ty = &field.ty;

        // An explicit `instructions` wins over the doc comment.
        let instructions = match instructions {
            Some(instructions) => {
                not_blank(&instructions, "instructions must not be blank")?;
                instructions
            }
            None => attr::doc_comment(&field.attrs).ok_or_else(|| {
                syn::Error::new_spanned(
                    ident,
                    "missing instructions: add a doc comment or \
                     `#[decision(instructions = \"...\")]`",
                )
            })?,
        };
        let described = yes.is_some() || no.is_some();
        let yes = attr::description(yes.as_ref())?;
        let no = attr::description(no.as_ref())?;

        let name = match &rename {
            Some(rename) => {
                not_blank(rename, "question names must not be blank")?;
                rename.value()
            }
            None => ident.unraw().to_string(),
        };
        if !names.insert(name.clone()) {
            return Err(syn::Error::new_spanned(
                field,
                format!("duplicate question name {name:?}"),
            ));
        }

        // Every use names `Field<#ty>` with the field type's span, so a type
        // that is not a decision field is reported once, at the field.
        let field_krate = respan(krate.clone(), ty.span());
        let field_ty =
            quote_spanned! {ty.span()=> #field_krate::decision::__private::Field::<#ty> };
        let generic_ty = mentions_any(ty, &generic_names);
        if generic_ty {
            generic_fields.push((ty, field_krate.clone()));
        }
        if !generic_ty {
            // The one place a wrong field type is reported; the uses below
            // resolve through `FieldFallback` instead of erroring again.
            // Generic field types are checked by the impl's where clause.
            assertions.push(quote_spanned! {ty.span()=>
                const _: () = #field_krate::decision::__private::assert_field::<#ty>();
            });
        }
        let mut question = quote_spanned! {ty.span()=>
            #field_ty::question(#instructions, #yes, #no)
        };
        if described {
            let assertion = quote_spanned! {ty.span()=>
                ::core::assert!(
                    #field_ty::IS_YES_NO,
                    "`yes` and `no` describe the outcomes of yes/no questions only",
                )
            };
            if generic_ty {
                // Checked where the field type is known: when the impl is
                // instantiated.
                question = quote_spanned! {ty.span()=> { const { #assertion; } #question } };
            } else {
                // Item-level consts are checked by `cargo check`.
                assertions.push(quote_spanned! {ty.span()=>
                    const _: () = {
                        #[allow(unused_imports)]
                        use #field_krate::decision::__private::FieldFallback as _;
                        #assertion;
                    };
                });
            }
        }
        questions.push(quote! { (::std::string::String::from(#name), #question) });
        decoders.push(quote_spanned! {ty.span()=>
            #ident: #field_ty::answer(#response, #name)?
        });
    }

    if !generic_fields.is_empty() {
        let where_clause = input.generics.make_where_clause();
        for (ty, field_krate) in generic_fields {
            // Spanned at the field, so an unsatisfied bound points there.
            where_clause.predicates.push(syn::parse2(
                quote_spanned! {ty.span()=> #ty: #field_krate::decision::DecisionField },
            )?);
        }
    }

    let name = &input.ident;
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    Ok(quote! {
        #(#assertions)*

        impl #impl_generics #krate::decision::Decision for #name #ty_generics #where_clause {
            fn questions() -> ::std::vec::Vec<(::std::string::String, #krate::decision::Question)> {
                #[allow(unused_imports)]
                use #krate::decision::__private::FieldFallback as _;
                ::std::vec![#(#questions),*]
            }

            fn from_response(
                #response: &#krate::DecisionResponse,
            ) -> #krate::Result<Self> {
                #[allow(unused_imports)]
                use #krate::decision::__private::FieldFallback as _;
                ::core::result::Result::Ok(Self {
                    #(#decoders,)*
                })
            }
        }
    })
}

fn not_blank(lit: &LitStr, message: &str) -> syn::Result<()> {
    if lit.value().trim().is_empty() {
        return Err(syn::Error::new_spanned(lit, message));
    }
    Ok(())
}

/// `tokens` with every span set to `span`, so diagnostics about an expression
/// starting with them point at `span`.
fn respan(tokens: TokenStream, span: Span) -> TokenStream {
    tokens
        .into_iter()
        .map(|mut token| {
            if let TokenTree::Group(group) = &token {
                let mut respanned =
                    proc_macro2::Group::new(group.delimiter(), respan(group.stream(), span));
                respanned.set_span(span);
                token = TokenTree::Group(respanned);
            }
            token.set_span(span);
            token
        })
        .collect()
}

/// Whether `ty` mentions one of the struct's type parameters.
fn mentions_any(ty: &Type, params: &HashSet<String>) -> bool {
    fn walk(tokens: TokenStream, params: &HashSet<String>) -> bool {
        tokens.into_iter().any(|token| match token {
            TokenTree::Ident(ident) => params.contains(&ident.to_string()),
            TokenTree::Group(group) => walk(group.stream(), params),
            _ => false,
        })
    }
    !params.is_empty() && walk(quote!(#ty), params)
}
