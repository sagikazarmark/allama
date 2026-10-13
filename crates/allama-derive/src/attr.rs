use std::collections::HashSet;

use heck::{ToKebabCase, ToLowerCamelCase, ToShoutySnakeCase, ToSnakeCase, ToUpperCamelCase};
use proc_macro2::TokenStream;
use quote::quote;
use syn::ext::IdentExt;
use syn::meta::ParseNestedMeta;
use syn::spanned::Spanned;
use syn::{
    Attribute, Data, DeriveInput, Expr, ExprLit, Fields, Ident, Lit, LitStr, Meta, Path, Variant,
};

/// Calls `f` for each item in every `#[decision(...)]` attribute.
pub fn parse(
    attrs: &[Attribute],
    mut f: impl FnMut(ParseNestedMeta) -> syn::Result<()>,
) -> syn::Result<()> {
    for attr in attrs.iter().filter(|attr| attr.path().is_ident("decision")) {
        attr.parse_nested_meta(&mut f)?;
    }
    Ok(())
}

/// Parses `name = "..."` into `slot`, rejecting duplicates.
pub fn set(slot: &mut Option<LitStr>, meta: &ParseNestedMeta) -> syn::Result<()> {
    if slot.is_some() {
        return Err(meta.error("duplicate attribute"));
    }
    *slot = Some(meta.value()?.parse()?);
    Ok(())
}

pub fn unknown(meta: &ParseNestedMeta, expected: &str) -> syn::Error {
    let name = meta
        .path
        .get_ident()
        .map(Ident::to_string)
        .unwrap_or_default();
    meta.error(format!("unknown attribute `{name}`, expected {expected}"))
}

/// The path to `allama` from `crate = "..."`, default `::allama`.
pub fn crate_path(lit: Option<LitStr>) -> syn::Result<TokenStream> {
    match lit {
        Some(lit) => {
            let path: Path = lit.parse().map_err(|_| {
                syn::Error::new_spanned(
                    &lit,
                    "`crate` must be a path to `allama`, e.g. `crate = \"my_facade::allama\"`",
                )
            })?;
            Ok(quote!(#path))
        }
        None => Ok(quote!(::allama)),
    }
}

/// The first paragraph of the item's doc comment as one line: each line
/// trimmed and joined with spaces, up to the first blank line after text.
/// Later paragraphs are for Rust readers (details, examples, rustdoc
/// links) and never reach the model. `None` when there is no doc comment
/// or it is blank. `#[doc = ...]` attributes whose value is not a string
/// literal, such as `include_str!(...)`, are skipped.
pub fn doc_comment(attrs: &[Attribute]) -> Option<LitStr> {
    let mut span = None;
    let mut lines = Vec::new();
    'attrs: for attr in attrs.iter().filter(|attr| attr.path().is_ident("doc")) {
        let Meta::NameValue(meta) = &attr.meta else {
            continue;
        };
        let Expr::Lit(ExprLit {
            lit: Lit::Str(lit), ..
        }) = &meta.value
        else {
            continue;
        };
        let value = lit.value();
        // A bare `///` is an attribute with an empty value, which has no
        // lines: it is one blank line.
        let blank = value.is_empty().then_some("");
        let block = is_starred_block(&value);
        for line in value
            .lines()
            .map(|line| unstar(line.trim(), block))
            .chain(blank)
        {
            if !line.is_empty() {
                lines.push(line.to_owned());
                span.get_or_insert(attr.span());
            } else if !lines.is_empty() {
                break 'attrs;
            }
        }
    }
    Some(LitStr::new(&lines.join(" "), span?))
}

/// Whether `value` is a `/** ... */` block comment whose lines start with
/// `*`, as rustdoc reads them: every line with text starts with `*` once
/// trimmed.
fn is_starred_block(value: &str) -> bool {
    let mut lines = value.lines().map(str::trim).filter(|line| !line.is_empty());
    value.contains('\n')
        && lines.clone().next().is_some()
        && lines.all(|line| line.starts_with('*'))
}

/// `line` without the leading `*` of a starred block comment (and the space
/// after it), as rustdoc strips it.
fn unstar(line: &str, block: bool) -> &str {
    match line.strip_prefix('*') {
        Some(rest) if block => rest.trim_start(),
        _ => line,
    }
}

/// Rejects a description that is present but blank.
pub fn description(lit: Option<&LitStr>) -> syn::Result<TokenStream> {
    match lit {
        Some(lit) if lit.value().trim().is_empty() => Err(syn::Error::new_spanned(
            lit,
            "descriptions must not be blank",
        )),
        Some(lit) => Ok(quote!(::core::option::Option::Some(#lit))),
        None => Ok(quote!(::core::option::Option::None)),
    }
}

/// Returns the variants of a non-generic enum of at least two unit variants.
pub fn unit_variants<'a>(input: &'a DeriveInput, derive: &str) -> syn::Result<Vec<&'a Variant>> {
    let Data::Enum(data) = &input.data else {
        return Err(syn::Error::new_spanned(
            &input.ident,
            format!("`{derive}` can only be derived for enums"),
        ));
    };

    if !input.generics.params.is_empty() {
        return Err(syn::Error::new_spanned(
            &input.generics,
            format!("`{derive}` cannot be derived for generic enums"),
        ));
    }

    for variant in &data.variants {
        if !matches!(variant.fields, Fields::Unit) {
            return Err(syn::Error::new_spanned(
                variant,
                format!("`{derive}` variants cannot have fields: the model only picks a variant"),
            ));
        }
    }

    let count = data.variants.len();
    if count < 2 {
        return Err(syn::Error::new_spanned(
            &input.ident,
            format!("`{derive}` enums need at least two variants, found {count}"),
        ));
    }

    Ok(data.variants.iter().collect())
}

/// One variant of a `ChoiceSet` or `Scale` enum: its name on the wire
/// (key or label) and its description.
pub struct Entry<'a> {
    pub variant: &'a Variant,
    pub name: String,
    pub description: TokenStream,
}

/// Parses the container's `rename_all` and `crate`, and each variant's
/// `rename` and `description` (falling back to its doc comment), checking that names are non-blank and unique.
pub fn entries<'a>(
    input: &'a DeriveInput,
    derive: &str,
    what: &str,
) -> syn::Result<(Vec<Entry<'a>>, TokenStream)> {
    let variants = unit_variants(input, derive)?;

    let (mut rename_all, mut krate) = (None, None);
    parse(&input.attrs, |meta| {
        if meta.path.is_ident("rename_all") {
            set(&mut rename_all, &meta)
        } else if meta.path.is_ident("crate") {
            set(&mut krate, &meta)
        } else {
            Err(unknown(&meta, "`rename_all` or `crate`"))
        }
    })?;
    let rename_all = rename_all.map(RenameAll::parse).transpose()?;

    let mut names = HashSet::new();
    let mut entries = Vec::new();
    for variant in variants {
        let (mut rename, mut description) = (None, None);
        parse(&variant.attrs, |meta| {
            if meta.path.is_ident("rename") {
                set(&mut rename, &meta)
            } else if meta.path.is_ident("description") {
                set(&mut description, &meta)
            } else {
                Err(unknown(&meta, "`rename` or `description`"))
            }
        })?;

        let name = match &rename {
            Some(rename) => rename.value(),
            None => {
                let name = variant.ident.unraw().to_string();
                rename_all.map_or(name.clone(), |rename_all| rename_all.apply(&name))
            }
        };
        if name.trim().is_empty() {
            return Err(match &rename {
                Some(rename) => {
                    syn::Error::new_spanned(rename, format!("{what}s must not be blank"))
                }
                None => syn::Error::new_spanned(
                    variant,
                    format!("this variant's {what} is blank after `rename_all`; add `rename`"),
                ),
            });
        }
        if !names.insert(name.clone()) {
            return Err(syn::Error::new_spanned(
                variant,
                format!("duplicate {what} {name:?}"),
            ));
        }

        entries.push(Entry {
            variant,
            name,
            // An explicit `description` wins over the doc comment.
            description: self::description(
                description.or_else(|| doc_comment(&variant.attrs)).as_ref(),
            )?,
        });
    }

    Ok((entries, crate_path(krate)?))
}

#[derive(Clone, Copy)]
enum RenameAll {
    Lower,
    Upper,
    Snake,
    Kebab,
    ScreamingSnake,
    Camel,
    Pascal,
}

impl RenameAll {
    fn parse(lit: LitStr) -> syn::Result<Self> {
        Ok(match lit.value().as_str() {
            "lowercase" => Self::Lower,
            "UPPERCASE" => Self::Upper,
            "snake_case" => Self::Snake,
            "kebab-case" => Self::Kebab,
            "SCREAMING_SNAKE_CASE" => Self::ScreamingSnake,
            "camelCase" => Self::Camel,
            "PascalCase" => Self::Pascal,
            _ => {
                return Err(syn::Error::new_spanned(
                    lit,
                    "expected one of `lowercase`, `UPPERCASE`, `snake_case`, `kebab-case`, \
                     `SCREAMING_SNAKE_CASE`, `camelCase`, `PascalCase`",
                ));
            }
        })
    }

    fn apply(self, name: &str) -> String {
        match self {
            Self::Lower => name.to_lowercase(),
            Self::Upper => name.to_uppercase(),
            Self::Snake => name.to_snake_case(),
            Self::Kebab => name.to_kebab_case(),
            Self::ScreamingSnake => name.to_shouty_snake_case(),
            Self::Camel => name.to_lower_camel_case(),
            Self::Pascal => name.to_upper_camel_case(),
        }
    }
}

#[cfg(test)]
mod tests {
    use syn::parse_quote;

    use super::*;

    fn doc(attrs: &[Attribute]) -> Option<String> {
        doc_comment(attrs).map(|lit| lit.value())
    }

    #[test]
    fn doc_comment_joins_the_first_paragraph_with_spaces() {
        let attrs: Vec<Attribute> = vec![
            parse_quote!(#[doc = " No time"]),
            parse_quote!(#[doc = "   pressure. "]),
            parse_quote!(#[doc = ""]),
            parse_quote!(#[doc = " See [`Level`] for the scale."]),
        ];
        assert_eq!(doc(&attrs).as_deref(), Some("No time pressure."));
    }

    #[test]
    fn doc_comment_skips_leading_blank_lines() {
        let attrs: Vec<Attribute> = vec![
            parse_quote!(#[doc = ""]),
            parse_quote!(#[doc = "   "]),
            parse_quote!(#[doc = " First"]),
            parse_quote!(#[doc = " second"]),
        ];
        assert_eq!(doc(&attrs).as_deref(), Some("First second"));
    }

    #[test]
    fn doc_comment_splits_block_comments_into_lines() {
        let attrs: Vec<Attribute> = vec![parse_quote!(#[doc = "\n  First\n  second\n"])];
        assert_eq!(doc(&attrs).as_deref(), Some("First second"));
        // A blank line inside a block comment ends the paragraph too.
        let attrs: Vec<Attribute> = vec![parse_quote!(
            #[doc = "\n  First\n  second\n\n  ```\n  example();\n  ```\n"]
        )];
        assert_eq!(doc(&attrs).as_deref(), Some("First second"));
    }

    /// The `*` of a starred block comment never
    /// reaches the model, as rustdoc strips it.
    #[test]
    fn doc_comment_strips_the_stars_of_a_block_comment() {
        let attrs: Vec<Attribute> = vec![parse_quote!(
            #[doc = "\n * Which label fits?\n * Pick one.\n *\n * Details for readers.\n "]
        )];
        assert_eq!(doc(&attrs).as_deref(), Some("Which label fits? Pick one."));
        // A `///` line that starts a Markdown list keeps its `*`.
        let attrs: Vec<Attribute> = vec![parse_quote!(#[doc = " * a list item"])];
        assert_eq!(doc(&attrs).as_deref(), Some("* a list item"));
        // So does a block whose lines are not all starred.
        let attrs: Vec<Attribute> = vec![parse_quote!(#[doc = "\n  Pick:\n  * one\n"])];
        assert_eq!(doc(&attrs).as_deref(), Some("Pick: * one"));
    }

    #[test]
    fn doc_comment_ignores_other_attributes() {
        let attrs: Vec<Attribute> = vec![
            parse_quote!(#[doc(hidden)]),
            parse_quote!(#[doc = include_str!("x.md")]),
            parse_quote!(#[decision(description = "x")]),
            parse_quote!(#[doc = " Kept"]),
        ];
        assert_eq!(doc(&attrs).as_deref(), Some("Kept"));
    }

    #[test]
    fn doc_comment_absent_or_blank() {
        assert_eq!(doc(&[]), None);
        let attrs: Vec<Attribute> = vec![parse_quote!(#[doc = ""]), parse_quote!(#[doc = "   "])];
        assert_eq!(doc(&attrs), None);
    }
}
