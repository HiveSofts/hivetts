//! Text preprocessing for natural, reliable synthesis.
//!
//! These helpers run before text reaches the speech engine so that spacing,
//! punctuation, zero-width characters, and symbol layout do not force the
//! engine into robotic or mispronounced readings. Number, date, and symbol
//! pronunciation is still delegated to the engine (eSpeak NG / Piper), which
//! understands those tokens natively; this module only guarantees the tokens
//! arrive cleanly separated and correctly spaced.

/// Writing direction inferred from the first strong character in the text.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Direction {
    Ltr,
    Rtl,
}

impl Direction {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Ltr => "ltr",
            Self::Rtl => "rtl",
        }
    }
}

const CLAUSE_DELIMITERS: [char; 5] = [',', ';', ':', '،', '؛'];
const SENTENCE_DELIMITERS: [char; 7] = ['.', '!', '?', '۔', '؟', '।', '॥'];
const CLOSING_CHARS: [char; 5] = ['"', ')', ']', '}', '\''];

fn is_punctuation(character: char) -> bool {
    character == '…'
        || CLAUSE_DELIMITERS.contains(&character)
        || SENTENCE_DELIMITERS.contains(&character)
}

/// Clean text for synthesis: drop stray zero-width marks (keeping the Persian
/// zero-width non-joiner, which is meaningful), normalize quotes and
/// ellipses, collapse redundant spaces, and enforce consistent spacing
/// around punctuation.
pub fn normalize(text: &str) -> String {
    let stripped = strip_unwanted(text);
    let symbols = replace_symbols(&stripped);
    let collapsed = collapse_spaces(&symbols);
    let spaced = fix_punctuation_spacing(&collapsed);
    spaced.trim().to_string()
}

/// Remove the BOM and the joiner/zero-width-space marks that leak in from
/// copied documents, normalize line endings, and turn tabs into spaces. The
/// ZWNJ (`U+200C`) is preserved because Persian orthography relies on it.
fn strip_unwanted(text: &str) -> String {
    text.chars()
        .filter(|character| {
            !matches!(character, '\u{FEFF}' | '\u{200B}' | '\u{200D}' | '\r')
        })
        .map(|character| if character == '\t' { ' ' } else { character })
        .collect()
}

/// Normalize typographic quotes to their ASCII equivalents and collapse three
/// literal dots into a single ellipsis so the engine pauses once, not thrice.
fn replace_symbols(text: &str) -> String {
    text.replace("...", "…")
        .replace(['“', '”'], "\"")
        .replace(['‘', '’'], "'")
}

/// Collapse runs of two or more spaces into a single space, leaving paragraph
/// breaks untouched.
fn collapse_spaces(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let mut in_spaces = false;
    for character in text.chars() {
        if character == ' ' {
            if !in_spaces {
                result.push(' ');
                in_spaces = true;
            }
        } else {
            if character == '\n' {
                // Drop spaces that trailed a line so wrapping never lags.
                while result.ends_with(' ') {
                    result.pop();
                }
            }
            result.push(character);
            in_spaces = false;
        }
    }
    result
}

/// Ensure no space precedes a punctuation mark and exactly one space follows
/// it, while protecting numeric separators (`1,000`, `3.14`, `12/05/2026`)
/// and collapsing duplicated marks (`!!`, `,,`) into one.
fn fix_punctuation_spacing(text: &str) -> String {
    let characters: Vec<char> = text.chars().collect();
    let mut output: Vec<char> = Vec::with_capacity(characters.len());
    let mut index = 0;
    while index < characters.len() {
        let character = characters[index];
        if !is_punctuation(character) {
            output.push(character);
            index += 1;
            continue;
        }
        // Repeated identical punctuation collapses to a single mark.
        if output.last() == Some(&character) {
            index += 1;
            continue;
        }
        // A comma or period wedged between digits is a number separator.
        let numeric_separator = (character == ',' || character == '.')
            && output.last().is_some_and(|previous| previous.is_ascii_digit())
            && characters.get(index + 1).is_some_and(|next| next.is_ascii_digit());
        if numeric_separator {
            output.push(character);
            index += 1;
            continue;
        }
        while output.last() == Some(&' ') {
            output.pop();
        }
        output.push(character);
        index += 1;
        while characters.get(index) == Some(&' ') {
            index += 1;
        }
        if let Some(&next) = characters.get(index) {
            if next != '\n' && !is_punctuation(next) && !CLOSING_CHARS.contains(&next) {
                output.push(' ');
            }
        }
    }
    output.into_iter().collect()
}

/// Detect the dominant writing direction by inspecting the first strong
/// character: RTL scripts (Hebrew, Arabic/Persian, and their presentation
/// forms) read right-to-left; everything else defaults to left-to-right.
pub fn detect_direction(text: &str) -> Direction {
    for character in text.chars() {
        let code_point = character as u32;
        let rtl = (0x0590..=0x05FF).contains(&code_point)
            || (0x0600..=0x06FF).contains(&code_point)
            || (0x0750..=0x077F).contains(&code_point)
            || (0x08A0..=0x08FF).contains(&code_point)
            || (0xFB1D..=0xFDFF).contains(&code_point)
            || (0xFE70..=0xFEFF).contains(&code_point);
        if rtl {
            return Direction::Rtl;
        }
        if character.is_alphabetic() {
            return Direction::Ltr;
        }
    }
    Direction::Ltr
}

/// Best-effort ISO 639-1 language code for automatic language detection.
/// Returns `None` for empty or unclassifiable text so the caller can fall
/// back to the configured language. `whatlang` reports ISO 639-3 codes, so
/// the common ones are narrowed to the two-letter codes the engines accept.
pub fn detect_language(text: &str) -> Option<String> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return None;
    }
    whatlang::detect_lang(trimmed)
        .map(|language| iso639_1(language.code()).to_string())
}

/// Narrow frequently detected ISO 639-3 codes to ISO 639-1 for eSpeak NG and
/// Piper. Unknown codes are returned unchanged so the caller still receives a
/// best-effort result.
fn iso639_1(code: &str) -> &str {
    match code {
        "eng" => "en",
        "fas" | "pes" | "prs" => "fa",
        "ara" => "ar",
        "tur" => "tr",
        "deu" => "de",
        "fra" => "fr",
        "spa" => "es",
        "por" => "pt",
        "ita" => "it",
        "rus" => "ru",
        "nld" => "nl",
        "pol" => "pl",
        "ukr" => "uk",
        "heb" => "he",
        "urd" => "ur",
        "hin" => "hi",
        "ben" => "bn",
        "aze" => "az",
        "ell" => "el",
        other => other,
    }
}

/// Split text into display words for the read-along timeline and highlight
/// model, preserving any punctuation attached to each token.
pub fn segment_words(text: &str) -> Vec<String> {
    text.split_whitespace().map(str::to_string).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_strips_zero_width_but_keeps_zwnj() {
        // ZWSP (200B) and ZWJ (200D) removed; ZWNJ (200C) preserved.
        let input = "ن\u{200C}می\u{200B}رم\u{200D}";
        let result = normalize(input);
        assert!(result.contains('\u{200C}'));
        assert!(!result.contains('\u{200B}'));
        assert!(!result.contains('\u{200D}'));
    }

    #[test]
    fn normalize_removes_bom_and_crlf() {
        assert_eq!(normalize("\u{FEFF}Hello\r\nWorld"), "Hello\nWorld");
    }

    #[test]
    fn normalize_fixes_spacing_and_collapses_punctuation() {
        assert_eq!(normalize("Hello ,  world ."), "Hello, world.");
        assert_eq!(normalize("Wow!!  Really?"), "Wow! Really?");
        assert_eq!(normalize("Wait ... there"), "Wait… there");
    }

    #[test]
    fn normalize_protects_numbers_and_dates() {
        assert_eq!(normalize("1,000 dollars"), "1,000 dollars");
        assert_eq!(normalize("pi is 3.14 ."), "pi is 3.14.");
    }

    #[test]
    fn normalize_keeps_quoted_sentence_intact() {
        assert_eq!(normalize("He said “hi”."), "He said \"hi\".");
    }

    #[test]
    fn direction_detects_latin_and_arabic() {
        assert_eq!(detect_direction("Hello world"), Direction::Ltr);
        assert_eq!(detect_direction("مرحبا"), Direction::Rtl);
        assert_eq!(detect_direction("سلام"), Direction::Rtl);
        assert_eq!(detect_direction("123 abc"), Direction::Ltr);
    }

    #[test]
    fn language_detection_returns_iso_codes() {
        assert_eq!(
            detect_language("This is an English sentence about testing."),
            Some("en".into())
        );
        assert_eq!(detect_language(""), None);
    }

    #[test]
    fn segment_words_splits_on_whitespace() {
        assert_eq!(
            segment_words("Two  little   words."),
            vec!["Two".to_string(), "little".to_string(), "words.".to_string()]
        );
    }
}
