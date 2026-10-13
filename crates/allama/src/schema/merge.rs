//! The sibling `$ref` merge: [`merge_sibling_refs`].

use serde_json::{Map, Value};

use super::{ANNOTATION_KEYWORDS, DATA_KEYWORDS, SchemaError, StrictSchemaError, resolve};

/// Flatten every `$ref` that has sibling keywords into one object schema.
///
/// In JSON Schema 2020-12, `{"$ref": "#/$defs/A", "properties": {..}}`
/// means "valid against `A` *and* against the siblings", like an `allOf` of
/// the two. schemars writes this shape for an internally tagged newtype
/// variant (the inner struct's `$ref` beside the tag property) and for a
/// documented field (a `description` beside the `$ref`). Grammar engines
/// read it differently: Google's `responseJsonSchema` forbids siblings next
/// to `$ref`, OpenAI rejects `allOf`, and a strict pass that closes objects
/// makes the two halves contradict each other. This pass replaces each such
/// node with a single object:
///
/// - the target's keywords and the siblings' are combined. `properties` and
///   `required` are unioned; for annotations (`title`, `description`, …)
///   the sibling wins; any other keyword must agree on both sides;
/// - no `allOf` is produced, and a `$ref` without siblings stays a
///   reference, so recursion through bare references is kept;
/// - targets are read from `schema` as given, before any rewrite, so a later
///   pass (closing objects, say) cannot turn a valid merge into a conflict.
///
/// A node that cannot be flattened is left as it is and reported in
/// [`SiblingRefMerge::unmerged`]: two sides that constrain one keyword or
/// one property differently, a key one side closes
/// (`additionalProperties: false`) and the other uses, or a reference with
/// constraining siblings that leads back to itself. A strict dialect treats
/// any of those as not representable; an engine with no strict mode may
/// send the node unmerged and report it. A recursive reference whose
/// siblings are annotations only becomes a bare reference: the annotations
/// change nothing the schema accepts.
///
/// ```
/// use allama::adapter::schema::merge_sibling_refs;
/// use serde_json::json;
///
/// // An internally tagged newtype variant, as schemars writes it.
/// let schema = json!({
///     "$ref": "#/$defs/Inner",
///     "type": "object",
///     "properties": {"kind": {"type": "string", "const": "New"}},
///     "required": ["kind"],
///     "$defs": {"Inner": {
///         "type": "object",
///         "properties": {"a": {"type": "integer"}},
///         "required": ["a"]
///     }}
/// });
/// let merged = merge_sibling_refs(&schema)?;
/// assert!(merged.unmerged.is_empty());
/// assert_eq!(merged.schema["properties"]["a"], json!({"type": "integer"}));
/// assert_eq!(merged.schema["required"], json!(["a", "kind"]));
/// assert!(merged.schema.get("$ref").is_none());
/// # Ok::<(), allama::adapter::schema::SchemaError>(())
/// ```
///
/// # Errors
///
/// Returns [`SchemaError::UnresolvedRef`] when a `$ref` with siblings points
/// outside the document or to a missing definition.
pub fn merge_sibling_refs(schema: &Value) -> Result<SiblingRefMerge, SchemaError> {
    let mut merger = Merger {
        document: schema,
        unmerged: Vec::new(),
    };
    let mut out = schema.clone();
    merger.walk(&mut out, "", &mut Vec::new())?;
    Ok(SiblingRefMerge {
        schema: out,
        unmerged: merger.unmerged,
    })
}

/// The result of [`merge_sibling_refs`].
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct SiblingRefMerge {
    /// The schema with every mergeable `$ref` and its siblings flattened.
    pub schema: Value,
    /// The nodes left unmerged, in document order.
    pub unmerged: Vec<UnmergedRef>,
}

impl SiblingRefMerge {
    /// The merged schema, for a strict dialect: a node left unmerged is
    /// [`StrictSchemaError::NotRepresentable`], naming the first one.
    ///
    /// A grammar that closes objects cannot express a `$ref` whose target
    /// conflicts with its siblings. An engine with no strict mode reads
    /// [`unmerged`](Self::unmerged) instead, and may send those nodes as they
    /// are.
    ///
    /// ```
    /// use allama::adapter::schema::{StrictSchemaError, merge_sibling_refs};
    /// use serde_json::json;
    ///
    /// let conflict = json!({
    ///     "type": "object",
    ///     "properties": {"v": {"$ref": "#/$defs/S", "type": "integer"}},
    ///     "$defs": {"S": {"type": "string"}}
    /// });
    /// let err = merge_sibling_refs(&conflict)?.into_strict().unwrap_err();
    /// assert!(matches!(err, StrictSchemaError::NotRepresentable { reason, .. } if reason.contains("/properties/v")));
    /// # Ok::<(), StrictSchemaError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// [`StrictSchemaError::NotRepresentable`] when a node was left
    /// unmerged.
    pub fn into_strict(self) -> Result<Value, StrictSchemaError> {
        match self.unmerged.first() {
            Some(unmerged) => Err(StrictSchemaError::not_representable(unmerged.to_string())),
            None => Ok(self.schema),
        }
    }
}

/// A `$ref` with sibling keywords that [`merge_sibling_refs`] could not
/// flatten.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct UnmergedRef {
    /// The JSON Pointer of the node (`/` for the root).
    pub path: String,
    /// Why it cannot be flattened.
    pub reason: String,
}

impl std::fmt::Display for UnmergedRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "the `$ref` with sibling keywords at `{}` cannot be merged: {}",
            self.path, self.reason
        )
    }
}

/// Keywords of a `$ref` target that only make sense at the document root.
const ROOT_ONLY_KEYWORDS: &[&str] = &["$schema", "$id", "$anchor", "$defs", "definitions"];

/// Keywords whose value maps names to subschemas.
const SCHEMA_MAP_KEYWORDS: &[&str] = &[
    "properties",
    "patternProperties",
    "$defs",
    "definitions",
    "dependentSchemas",
];

struct Merger<'a> {
    /// The document targets are read from: the caller's schema, unchanged.
    document: &'a Value,
    unmerged: Vec<UnmergedRef>,
}

/// Why one node could not be flattened.
enum Failure {
    Conflict(String),
    Invalid(SchemaError),
}

impl Merger<'_> {
    /// Merge the nodes under `node`. `stack` holds the references whose
    /// copied-in content is being walked, to stop at recursion.
    fn walk(
        &mut self,
        node: &mut Value,
        path: &str,
        stack: &mut Vec<String>,
    ) -> Result<(), SchemaError> {
        match node {
            Value::Object(map) => {
                let mut pushed = false;
                if let Some(Value::String(reference)) = map.get("$ref")
                    && map.len() > 1
                {
                    let reference = reference.clone();
                    match self.merge_node(map, &reference, stack) {
                        Ok(Some(flat)) => {
                            *map = flat;
                            stack.push(reference);
                            pushed = true;
                        }
                        Ok(None) => {}
                        Err(Failure::Conflict(reason)) => self.unmerged.push(UnmergedRef {
                            path: display_path(path).to_owned(),
                            reason,
                        }),
                        Err(Failure::Invalid(error)) => return Err(error),
                    }
                }
                let result = self.walk_children(map, path, stack);
                if pushed {
                    stack.pop();
                }
                result
            }
            Value::Array(items) => {
                for (index, item) in items.iter_mut().enumerate() {
                    self.walk(item, &format!("{path}/{index}"), stack)?;
                }
                Ok(())
            }
            _ => Ok(()),
        }
    }

    fn walk_children(
        &mut self,
        map: &mut Map<String, Value>,
        path: &str,
        stack: &mut Vec<String>,
    ) -> Result<(), SchemaError> {
        for (key, value) in map.iter_mut() {
            if DATA_KEYWORDS.contains(&key.as_str()) || key == "required" {
                continue;
            }
            let child = format!("{path}/{}", escape(key));
            if SCHEMA_MAP_KEYWORDS.contains(&key.as_str())
                && let Value::Object(entries) = value
            {
                for (name, entry) in entries.iter_mut() {
                    self.walk(entry, &format!("{child}/{}", escape(name)), stack)?;
                }
            } else {
                self.walk(value, &child, stack)?;
            }
        }
        Ok(())
    }

    /// The flattened form of `map`, a `$ref` with siblings. `None` keeps
    /// the node as it is (now a bare reference).
    fn merge_node(
        &self,
        map: &mut Map<String, Value>,
        reference: &str,
        stack: &[String],
    ) -> Result<Option<Map<String, Value>>, Failure> {
        let mut siblings = map.clone();
        siblings.remove("$ref");
        if stack.iter().any(|r| r == reference) {
            // Recursion: the target's content is already being walked, so
            // copying it in again would never end.
            if siblings
                .keys()
                .all(|k| ANNOTATION_KEYWORDS.contains(&k.as_str()))
            {
                map.retain(|key, _| key == "$ref");
                return Ok(None);
            }
            return Err(Failure::Conflict(format!(
                "`{reference}` refers back to itself, so its sibling keywords cannot be merged into it"
            )));
        }
        let mut seen = stack.to_vec();
        let target = self.flat_target(reference, &mut seen)?;
        combine(target, siblings)
            .map(Some)
            .map_err(Failure::Conflict)
    }

    /// The flattened content of the schema `reference` points to.
    fn flat_target(
        &self,
        reference: &str,
        seen: &mut Vec<String>,
    ) -> Result<Map<String, Value>, Failure> {
        if seen.iter().any(|r| r == reference) {
            return Err(Failure::Conflict(format!(
                "`{reference}` refers back to itself through `$ref`s with sibling keywords"
            )));
        }
        let target = resolve(self.document, reference)
            .ok_or_else(|| Failure::Invalid(SchemaError::UnresolvedRef(reference.to_owned())))?;
        let target = match target {
            Value::Bool(true) => return Ok(Map::new()),
            Value::Object(target) => target,
            _ => {
                return Err(Failure::Conflict(format!(
                    "`{reference}` refers to a schema that accepts no value"
                )));
            }
        };
        let mut own: Map<String, Value> = target
            .iter()
            .filter(|(key, _)| !ROOT_ONLY_KEYWORDS.contains(&key.as_str()))
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect();
        let Some(Value::String(next)) = own.remove("$ref") else {
            return Ok(own);
        };
        seen.push(reference.to_owned());
        let base = self.flat_target(&next, seen)?;
        seen.pop();
        combine(base, own).map_err(Failure::Conflict)
    }
}

/// Combine a `$ref` target's keywords with its siblings', as the `allOf` of
/// the two would constrain them.
fn combine(
    mut base: Map<String, Value>,
    siblings: Map<String, Value>,
) -> Result<Map<String, Value>, String> {
    check_closed(&base, &siblings, "the referenced schema", "its siblings")?;
    check_closed(
        &siblings,
        &base,
        "the sibling keywords",
        "the referenced schema",
    )?;
    for (key, value) in siblings {
        match key.as_str() {
            "properties" => {
                let Value::Object(theirs) = value else {
                    return Err("`properties` is not an object".into());
                };
                let ours = base
                    .entry("properties")
                    .or_insert_with(|| Value::Object(Map::new()));
                let Value::Object(ours) = ours else {
                    return Err("`properties` is not an object".into());
                };
                for (name, schema) in theirs {
                    match ours.get(&name) {
                        None => {
                            ours.insert(name, schema);
                        }
                        Some(existing) if *existing == schema => {}
                        Some(_) => {
                            return Err(format!(
                                "the property `{name}` is defined differently by the referenced schema and its siblings"
                            ));
                        }
                    }
                }
            }
            "required" => {
                let Value::Array(theirs) = value else {
                    return Err("`required` is not an array".into());
                };
                let ours = base
                    .entry("required")
                    .or_insert_with(|| Value::Array(Vec::new()));
                let Value::Array(ours) = ours else {
                    return Err("`required` is not an array".into());
                };
                for name in theirs {
                    if !ours.contains(&name) {
                        ours.push(name);
                    }
                }
            }
            annotation if ANNOTATION_KEYWORDS.contains(&annotation) => {
                base.insert(key, value);
            }
            _ => match base.get(&key) {
                None => {
                    base.insert(key, value);
                }
                Some(existing) if *existing == value => {}
                Some(_) => {
                    return Err(format!(
                        "`{key}` differs between the referenced schema and its siblings"
                    ));
                }
            },
        }
    }
    Ok(base)
}

/// Fail when `closed` forbids a key `other` declares or requires.
fn check_closed(
    closed: &Map<String, Value>,
    other: &Map<String, Value>,
    closed_name: &str,
    other_name: &str,
) -> Result<(), String> {
    if closed.get("additionalProperties") != Some(&Value::Bool(false))
        || closed.contains_key("patternProperties")
    {
        return Ok(());
    }
    let declared = closed.get("properties").and_then(Value::as_object);
    let used = other
        .get("properties")
        .and_then(Value::as_object)
        .into_iter()
        .flat_map(|p| p.keys().map(String::as_str))
        .chain(
            other
                .get("required")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .filter_map(Value::as_str),
        );
    for key in used {
        if !declared.is_some_and(|d| d.contains_key(key)) {
            return Err(format!(
                "{other_name} use the key `{key}`, which {closed_name} closes (`additionalProperties: false`) without declaring"
            ));
        }
    }
    Ok(())
}

fn display_path(path: &str) -> &str {
    if path.is_empty() { "/" } else { path }
}

fn escape(name: &str) -> String {
    name.replace('~', "~0").replace('/', "~1")
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn merged(schema: Value) -> SiblingRefMerge {
        merge_sibling_refs(&schema).unwrap()
    }

    #[test]
    fn internally_tagged_newtype_variants_are_flattened() {
        let schema = json!({
            "oneOf": [{
                "$ref": "#/$defs/Inner",
                "type": "object",
                "properties": {"kind": {"type": "string", "const": "New"}},
                "required": ["kind"]
            }],
            "$defs": {"Inner": {
                "type": "object",
                "properties": {"a": {"type": "integer"}, "b": {"type": ["string", "null"]}},
                "required": ["a"]
            }}
        });
        let out = merged(schema);
        assert!(out.unmerged.is_empty(), "{:?}", out.unmerged);
        assert_eq!(
            out.schema["oneOf"][0],
            json!({
                "type": "object",
                "properties": {
                    "a": {"type": "integer"},
                    "b": {"type": ["string", "null"]},
                    "kind": {"type": "string", "const": "New"}
                },
                "required": ["a", "kind"]
            })
        );
        // Definitions are kept for bare references elsewhere.
        assert!(out.schema["$defs"]["Inner"].is_object());
        assert!(!out.schema.to_string().contains("allOf"));
    }

    #[test]
    fn annotations_beside_a_ref_win() {
        let schema = json!({
            "type": "object",
            "properties": {"inner": {"$ref": "#/$defs/Inner", "description": "outer"}},
            "$defs": {"Inner": {"type": "object", "description": "inner", "properties": {}}}
        });
        assert_eq!(
            merged(schema).schema["properties"]["inner"],
            json!({"type": "object", "description": "outer", "properties": {}})
        );
    }

    #[test]
    fn bare_refs_and_recursion_through_them_are_kept() {
        let schema = json!({
            "$defs": {"Node": {
                "type": "object",
                "properties": {"children": {"type": "array", "items": {"$ref": "#/$defs/Node"}}}
            }},
            "type": "object",
            "properties": {"root": {"$ref": "#/$defs/Node", "description": "the root"}}
        });
        let out = merged(schema);
        assert!(out.unmerged.is_empty());
        assert_eq!(
            out.schema["properties"]["root"]["properties"]["children"]["items"],
            json!({"$ref": "#/$defs/Node"})
        );
        assert_eq!(
            out.schema["$defs"]["Node"]["properties"]["children"]["items"],
            json!({"$ref": "#/$defs/Node"})
        );
    }

    #[test]
    fn recursive_refs_with_annotations_become_bare_references() {
        let schema = json!({
            "$defs": {"Node": {
                "type": "object",
                "properties": {"next": {"$ref": "#/$defs/Node", "description": "the next node"}}
            }},
            "type": "object",
            "properties": {"start": {"$ref": "#/$defs/Node"}}
        });
        let out = merged(schema);
        assert!(out.unmerged.is_empty(), "{:?}", out.unmerged);
        // Unrolled once inside the definition, then a bare reference.
        assert_eq!(
            out.schema["$defs"]["Node"]["properties"]["next"]["properties"]["next"],
            json!({"$ref": "#/$defs/Node"})
        );
    }

    #[test]
    fn recursive_refs_with_constraining_siblings_are_unmerged() {
        let schema = json!({
            "$defs": {"Node": {
                "type": "object",
                "properties": {"next": {"$ref": "#/$defs/Node", "required": ["next"]}}
            }},
            "type": "object",
            "properties": {"start": {"$ref": "#/$defs/Node"}}
        });
        let out = merged(schema);
        assert_eq!(out.unmerged.len(), 1, "{:?}", out.unmerged);
        assert!(out.unmerged[0].reason.contains("refers back to itself"));
    }

    #[test]
    fn root_refs_with_defs_are_flattened() {
        let schema = json!({
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$ref": "#/$defs/A",
            "$defs": {"A": {"type": "object", "properties": {"x": {"type": "string"}}}}
        });
        let out = merged(schema);
        assert_eq!(out.schema["type"], "object");
        assert_eq!(out.schema["properties"]["x"]["type"], "string");
        assert!(out.schema.get("$ref").is_none());
        assert!(out.schema["$defs"]["A"].is_object());
    }

    #[test]
    fn chained_refs_are_followed() {
        let schema = json!({
            "properties": {"v": {"$ref": "#/$defs/A", "description": "v"}},
            "$defs": {
                "A": {"$ref": "#/$defs/B", "title": "A"},
                "B": {"type": "string", "title": "B"}
            }
        });
        assert_eq!(
            merged(schema).schema["properties"]["v"],
            json!({"type": "string", "title": "A", "description": "v"})
        );
    }

    #[test]
    fn conflicts_leave_the_node_unmerged() {
        let cases = [
            (
                json!({"$ref": "#/$defs/S", "type": "integer"}),
                "`type` differs",
            ),
            (
                json!({"$ref": "#/$defs/O", "properties": {"a": {"type": "integer"}}}),
                "property `a`",
            ),
            (
                json!({"$ref": "#/$defs/Closed", "properties": {"extra": {"type": "string"}}}),
                "`extra`",
            ),
            (
                json!({"$ref": "#/$defs/O", "additionalProperties": false, "properties": {}}),
                "`a`",
            ),
            (
                json!({"$ref": "#/$defs/Never", "title": "x"}),
                "accepts no value",
            ),
        ];
        for (node, needle) in cases {
            let schema = json!({
                "properties": {"field": node},
                "$defs": {
                    "S": {"type": "string"},
                    "O": {"type": "object", "properties": {"a": {"type": "string"}}},
                    "Closed": {"type": "object", "properties": {"a": {"type": "string"}}, "additionalProperties": false},
                    "Never": false
                }
            });
            let out = merged(schema.clone());
            assert_eq!(out.schema, schema, "{needle}");
            assert_eq!(out.unmerged.len(), 1, "{needle}");
            assert_eq!(out.unmerged[0].path, "/properties/field");
            assert!(
                out.unmerged[0].reason.contains(needle),
                "{needle}: {}",
                out.unmerged[0].reason
            );
        }
    }

    #[test]
    fn closed_targets_accept_siblings_they_declare() {
        let schema = json!({
            "properties": {"field": {"$ref": "#/$defs/Closed", "required": ["a"]}},
            "$defs": {"Closed": {"type": "object", "properties": {"a": {"type": "string"}}, "additionalProperties": false}}
        });
        let out = merged(schema);
        assert!(out.unmerged.is_empty());
        assert_eq!(out.schema["properties"]["field"]["required"], json!(["a"]));
    }

    #[test]
    fn unresolved_refs_with_siblings_are_errors() {
        assert_eq!(
            merge_sibling_refs(&json!({"$ref": "#/$defs/Missing", "title": "x"})),
            Err(SchemaError::UnresolvedRef("#/$defs/Missing".into()))
        );
        // A bare unresolved reference is left alone.
        let bare = json!({"properties": {"a": {"$ref": "#/$defs/Missing"}}});
        assert_eq!(merged(bare.clone()).schema, bare);
    }

    #[test]
    fn data_keywords_and_property_names_are_not_schemas() {
        let schema = json!({
            "properties": {"$ref": {"type": "string"}},
            "default": {"$ref": "#/$defs/A", "title": "data"},
            "$defs": {"A": {"type": "string"}}
        });
        assert_eq!(merged(schema.clone()).schema, schema);
    }
}
