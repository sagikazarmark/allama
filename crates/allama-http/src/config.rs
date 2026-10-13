//! Checking a provider builder's configuration: base URLs, secrets, and
//! the headers a provider sends with every request, and how a URL shows in
//! `Debug` output.
//!
//! Every bundled HTTP provider's `build()` (and `from_env()`) runs these
//! checks, so a malformed value fails there, with an
//! [authored](allama::Error::authored) [`allama::Error::Configuration`]
//! that names the setting or the environment variable, instead of failing
//! every request later as an invalid request. The messages never quote the
//! value: a base URL can carry a password, and a key is a secret. Provider
//! crates of your own use them the same way:
//!
//! ```
//! use allama_http::config;
//!
//! // A key read from a Kubernetes secret file keeps its trailing newline.
//! assert_eq!(config::secret("ACME_API_KEY", "sk-acme\n")?, "sk-acme");
//! assert_eq!(
//!     config::base_url("ACME_BASE_URL", " https://gw.example/v1/ ")?,
//!     "https://gw.example/v1"
//! );
//! let err = config::base_url("ACME_BASE_URL", "localhost:8080/v1").unwrap_err();
//! assert!(matches!(err, allama::Error::Configuration { .. }));
//! assert!(err.to_string().contains("ACME_BASE_URL"));
//! # Ok::<(), allama::Error>(())
//! ```
//!
//! Plain `http://` stays allowed: a local or LAN server (Ollama, vLLM) is a
//! common target. Only send keys over it to hosts you trust on that network.

use std::borrow::Cow;

use crate::{CredentialSlot, DEFAULT_USER_AGENT, Headers, is_credential_header};

/// Check and normalize a base URL from a builder setting or environment
/// variable named `setting`: surrounding whitespace and trailing slashes
/// are removed, and the URL must be an absolute `http` or `https` URL with
/// a host, no whitespace or control characters, and no query or fragment
/// (requests append their path to it).
///
/// A user name and password in the URL (`https://user:pass@gateway`) are
/// kept, since some gateways take them; [`redact_url`] hides them from
/// `Debug` output.
///
/// # Errors
///
/// An authored [`allama::Error::Configuration`] naming `setting`, never the
/// value.
pub fn base_url(setting: &str, value: &str) -> allama::Result<String> {
    let value = value.trim();
    let invalid = |reason: &str| {
        allama::Error::configuration(format!("`{setting}` is not a valid base URL: {reason}"))
            .authored()
    };
    if value.is_empty() {
        return Err(allama::Error::configuration(format!("`{setting}` is empty")).authored());
    }
    let Some((scheme, rest)) = value.split_once("://") else {
        return Err(invalid("it must start with `http://` or `https://`"));
    };
    if !scheme.eq_ignore_ascii_case("http") && !scheme.eq_ignore_ascii_case("https") {
        return Err(invalid("it must start with `http://` or `https://`"));
    }
    if value.chars().any(|c| c.is_whitespace() || c.is_control()) {
        return Err(invalid("it contains whitespace or control characters"));
    }
    if value.contains(['?', '#']) {
        return Err(invalid(
            "it has a query or fragment, and requests append their path to it",
        ));
    }
    let authority = rest.split('/').next().unwrap_or_default();
    let Some((host, port)) = split_authority(authority) else {
        return Err(invalid("it has no valid host"));
    };
    if host.is_empty() {
        return Err(invalid("it has no host"));
    }
    if port.is_some_and(|port| port.parse::<u16>().is_err()) {
        return Err(invalid("its port is not a number up to 65535"));
    }
    Ok(value.trim_end_matches('/').to_owned())
}

/// Check a secret (an API key or token) from a builder setting or
/// environment variable named `setting`, and return it without surrounding
/// whitespace: a key read from a file or a Kubernetes secret often ends in
/// a newline, which no HTTP header can carry.
///
/// # Errors
///
/// An authored [`allama::Error::Configuration`] naming `setting`, never the
/// value, when it is empty after trimming or holds a character an HTTP
/// header value cannot carry (anything but visible ASCII, spaces and tabs).
pub fn secret(setting: &str, value: &str) -> allama::Result<String> {
    let value = value.trim();
    if value.is_empty() {
        return Err(allama::Error::configuration(format!("`{setting}` is empty")).authored());
    }
    if !is_header_value(value) {
        return Err(allama::Error::configuration(format!(
            "`{setting}` contains characters an HTTP header cannot carry"
        ))
        .authored());
    }
    Ok(value.to_owned())
}

/// The headers a provider sends with every request: `User-Agent:`
/// [`DEFAULT_USER_AGENT`], then `headers` (a builder's
/// `header(name, value)` settings, in order), each replacing the headers of
/// the same name before it, so `header("user-agent", ..)` replaces the
/// default. Set them on each request with
/// [`HttpRequest::with_headers`](crate::HttpRequest::with_headers).
///
/// These headers belong to one provider: a gateway's routing header, a
/// tenant tag, a beta flag. Credentials do not: they are refused here, as
/// [`HttpRequest::apply_options`](crate::HttpRequest::apply_options) refuses
/// them in a request's options, so a key always goes through the builder's
/// key setting or a per-request [`Credential`](crate::Credential), where it
/// is scoped and redacted.
///
/// ```
/// use allama_http::{CredentialPlacement, CredentialSlot, DEFAULT_USER_AGENT, config};
///
/// let slot = CredentialSlot::new("acme", CredentialPlacement::Bearer);
/// let headers = config::static_headers([("x-route", "eu")], &slot)?;
/// assert_eq!(headers.get("user-agent"), Some(DEFAULT_USER_AGENT));
/// assert_eq!(headers.get("x-route"), Some("eu"));
///
/// let err = config::static_headers([("x-api-key", "sk-acme")], &slot).unwrap_err();
/// assert!(matches!(err, allama::Error::Configuration { .. }));
/// assert!(!err.to_string().contains("sk-acme"));
/// # Ok::<(), allama::Error>(())
/// ```
///
/// # Errors
///
/// An authored [`allama::Error::Configuration`] naming the header, never
/// its value, when the name is not a valid header name, the value holds a
/// character a header cannot carry (anything but visible ASCII, spaces and
/// tabs), or the header carries a credential: it is on the exact credential
/// list ([`is_credential_header`]), or `slot` writes or replaces it.
pub fn static_headers<I, N, V>(headers: I, slot: &CredentialSlot) -> allama::Result<Headers>
where
    I: IntoIterator<Item = (N, V)>,
    N: Into<String>,
    V: Into<String>,
{
    let mut all = Headers::new();
    all.insert("user-agent", DEFAULT_USER_AGENT);
    for (name, value) in headers {
        let (name, value) = (name.into(), value.into());
        if !is_header_name(&name) {
            return Err(allama::Error::configuration(format!(
                "`{name}` is not a valid HTTP header name"
            ))
            .authored());
        }
        if !is_header_value(&value) {
            return Err(allama::Error::configuration(format!(
                "the value of the header `{name}` contains characters an HTTP header cannot carry"
            ))
            .authored());
        }
        if is_credential_header(&name) || slot.owns_header(&name) {
            return Err(allama::Error::configuration(format!(
                "the header `{name}` carries a credential: configure the key with the builder's \
                 key setting, or send it with `HttpRequestOptions::credential`"
            ))
            .authored());
        }
        all.remove(&name);
        all.insert(name, value);
    }
    Ok(all)
}

/// `url` as `Debug` output and error messages may show it: without a user
/// name and password, query or fragment, which can carry credentials.
///
/// ```
/// use allama_http::config::redact_url;
///
/// assert_eq!(redact_url("https://user:pass@gw.example:8443/v1?key=k"), "https://gw.example:8443/v1");
/// assert_eq!(redact_url("https://api.example.com/v1"), "https://api.example.com/v1");
/// ```
#[must_use]
pub fn redact_url(url: &str) -> Cow<'_, str> {
    let (prefix, rest) = match url.find("://") {
        Some(at) => url.split_at(at + 3),
        None => ("", url),
    };
    let authority_end = rest.find(['/', '?', '#']).unwrap_or(rest.len());
    let userinfo_end = rest[..authority_end].rfind('@').map(|at| at + 1);
    let rest = &rest[userinfo_end.unwrap_or(0)..];
    let query = rest.find(['?', '#']);
    if userinfo_end.is_none() && query.is_none() {
        return Cow::Borrowed(url);
    }
    let rest = &rest[..query.unwrap_or(rest.len())];
    Cow::Owned(format!("{prefix}{rest}"))
}

/// Redact every URL in `text`, as [`redact_url`] does: for an HTTP
/// client's error message, which can quote the request URL.
#[cfg_attr(not(feature = "reqwest-middleware"), allow(dead_code))]
pub(crate) fn redact_urls_in(text: &str) -> Cow<'_, str> {
    if !text.contains("://") {
        return Cow::Borrowed(text);
    }
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(at) = rest.find("://") {
        let start = rest[..at]
            .rfind(|c: char| !(c.is_ascii_alphanumeric() || matches!(c, '+' | '-' | '.')))
            .map_or(0, |i| i + 1);
        let end = rest[at..]
            .find(|c: char| c.is_whitespace() || matches!(c, '"' | '\'' | '<' | '>' | ')' | '`'))
            .map_or(rest.len(), |i| at + i);
        out.push_str(&rest[..start]);
        out.push_str(&redact_url(&rest[start..end]));
        rest = &rest[end..];
    }
    out.push_str(rest);
    Cow::Owned(out)
}

/// The host a URL (an absolute or scheme-relative `Location`) points to,
/// for an authored message: lowercase, without user info or port. `None`
/// for a relative URL; a host with characters a host name cannot have is
/// reported as such, never quoted.
pub(crate) fn url_host(url: &str) -> Option<String> {
    let url = url.trim();
    let rest = match url.split_once("://") {
        Some((scheme, rest))
            if !scheme.is_empty()
                && scheme
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || matches!(c, '+' | '-' | '.')) =>
        {
            rest
        }
        _ => url.strip_prefix("//")?,
    };
    let authority = rest.split(['/', '?', '#']).next().unwrap_or_default();
    let host = split_authority(authority).map_or("", |(host, _)| host);
    let valid = !host.is_empty()
        && host.len() <= 253
        && host
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '-' | ':' | '[' | ']'));
    Some(if valid {
        host.to_ascii_lowercase()
    } else {
        "an invalid host".to_owned()
    })
}

/// An authority's host (an IPv6 literal keeps its brackets) and port,
/// without user info. `None` when brackets do not match.
fn split_authority(authority: &str) -> Option<(&str, Option<&str>)> {
    let host_port = authority.rsplit_once('@').map_or(authority, |(_, hp)| hp);
    if host_port.starts_with('[') {
        let close = host_port.find(']')?;
        let (host, after) = host_port.split_at(close + 1);
        return match after {
            "" => Some((host, None)),
            _ => Some((host, Some(after.strip_prefix(':')?))),
        };
    }
    if host_port.contains([']', '[']) {
        return None;
    }
    Some(match host_port.split_once(':') {
        Some((host, port)) => (host, Some(port)),
        None => (host_port, None),
    })
}

/// An HTTP header name: a non-empty token.
fn is_header_name(name: &str) -> bool {
    !name.is_empty()
        && name.bytes().all(|b| {
            b.is_ascii_alphanumeric()
                || matches!(
                    b,
                    b'!' | b'#'
                        | b'$'
                        | b'%'
                        | b'&'
                        | b'\''
                        | b'*'
                        | b'+'
                        | b'-'
                        | b'.'
                        | b'^'
                        | b'_'
                        | b'`'
                        | b'|'
                        | b'~'
                )
        })
}

/// A value every HTTP client sends as given: visible ASCII, spaces and
/// tabs.
fn is_header_value(value: &str) -> bool {
    value
        .bytes()
        .all(|b| b == b'\t' || (b' '..=b'~').contains(&b))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CredentialPlacement;

    #[test]
    fn base_urls_are_trimmed_and_checked() {
        for (value, expected) in [
            ("https://api.example.com/v1", "https://api.example.com/v1"),
            (
                "  http://localhost:11434/v1/  \n",
                "http://localhost:11434/v1",
            ),
            ("HTTPS://Gw.Example", "HTTPS://Gw.Example"),
            ("http://[::1]:8000/v1", "http://[::1]:8000/v1"),
            (
                "https://user:pass@gw.example/v1",
                "https://user:pass@gw.example/v1",
            ),
        ] {
            assert_eq!(base_url("base_url", value).unwrap(), expected, "{value}");
        }
        for value in [
            "",
            "  ",
            "localhost:8080/v1",
            "ftp://example.com",
            "https://",
            "https:///v1",
            "https://exa mple.com",
            "https://example.com/v1\n/x",
            "https://example.com/v1?api-version=1",
            "https://example.com/#x",
            "https://example.com:99999",
            "https://example.com:port",
            "http://[::1/v1",
            "https://attacker.example/x?.openai.azure.com/openai/v1",
        ] {
            let err = base_url("ACME_BASE_URL", value).unwrap_err();
            assert!(
                matches!(err, allama::Error::Configuration { .. }),
                "{value}: {err:?}"
            );
            let text = err.to_string();
            assert!(text.contains("`ACME_BASE_URL`"), "{text}");
            assert!(!err.is_retryable());
        }
        let err = base_url("u", "https://user:hunter2@x y").unwrap_err();
        assert!(!format!("{err} {err:?}").contains("hunter2"));
    }

    #[test]
    fn secrets_are_trimmed_and_must_fit_a_header() {
        assert_eq!(secret("KEY", " sk-x\n").unwrap(), "sk-x");
        assert_eq!(secret("KEY", "Bearer a b").unwrap(), "Bearer a b");
        for value in ["", " \n", "sk-\nx", "sk-é", "sk\0"] {
            let err = secret("OPENAI_API_KEY", value).unwrap_err();
            assert!(matches!(err, allama::Error::Configuration { .. }));
            assert!(err.to_string().contains("OPENAI_API_KEY"), "{err}");
            assert!(!err.to_string().contains("sk-"), "{err}");
        }
    }

    #[test]
    fn static_headers_start_with_the_user_agent_and_refuse_credentials() {
        let slot = CredentialSlot::new("acme", CredentialPlacement::Header("x-acme-key".into()))
            .replaces("x-acme-legacy");
        let headers = static_headers(Vec::<(String, String)>::new(), &slot).unwrap();
        assert_eq!(
            headers.iter().collect::<Vec<_>>(),
            [("user-agent", DEFAULT_USER_AGENT)]
        );
        assert!(DEFAULT_USER_AGENT.starts_with("allama/"));

        let headers = static_headers(
            [
                ("x-route", "a"),
                ("User-Agent", "my-app/1.0"),
                ("X-ROUTE", "b"),
                ("Idempotency-Key", "fine"),
            ],
            &slot,
        )
        .unwrap();
        assert_eq!(
            headers.iter().collect::<Vec<_>>(),
            [
                ("User-Agent", "my-app/1.0"),
                ("X-ROUTE", "b"),
                ("Idempotency-Key", "fine")
            ]
        );

        for (name, value) in [
            ("authorization", "Bearer SECRET"),
            ("x-api-key", "SECRET"),
            ("X-Acme-Key", "SECRET"),
            ("x-acme-legacy", "SECRET"),
            ("bad name", "v"),
            ("", "v"),
            ("x-ok", "SECRET\r\nx-injected: 1"),
        ] {
            let err = static_headers([(name, value)], &slot).unwrap_err();
            assert!(
                matches!(err, allama::Error::Configuration { .. }),
                "{name}: {err:?}"
            );
            assert!(!format!("{err} {err:?}").contains("SECRET"), "{err}");
        }
    }

    #[test]
    fn urls_are_redacted() {
        for (url, expected) in [
            ("https://api.example.com/v1", "https://api.example.com/v1"),
            ("https://user@gw.example/v1", "https://gw.example/v1"),
            ("https://user:pw@gw.example", "https://gw.example"),
            ("http://u:p@[::1]:80/v1?key=k#f", "http://[::1]:80/v1"),
            ("https://gw.example/v1/a@b", "https://gw.example/v1/a@b"),
            ("https://gw.example?key=k", "https://gw.example"),
        ] {
            assert_eq!(redact_url(url), expected, "{url}");
        }
        assert_eq!(
            redact_urls_in(
                "error sending request for url (https://u:pw@gw.example/v1?key=SECRET): reset"
            ),
            "error sending request for url (https://gw.example/v1): reset"
        );
        assert_eq!(redact_urls_in("no url here"), "no url here");
    }

    #[test]
    fn location_hosts() {
        for (location, host) in [
            ("https://Evil.Example/v1?token=x", Some("evil.example")),
            (
                "http://user:pw@169.254.169.254:80/latest",
                Some("169.254.169.254"),
            ),
            ("//cdn.example/x", Some("cdn.example")),
            ("/v1/messages", None),
            ("v1/messages", None),
            ("https://ev\u{1b}[31mil/", Some("an invalid host")),
        ] {
            assert_eq!(url_host(location).as_deref(), host, "{location}");
        }
    }
}
