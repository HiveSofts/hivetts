//! Plain-text extraction from the document formats the TTS page can open.
//!
//! TXT is read directly; PDF uses `pdf-extract`; DOCX (a zipped WordprocessingML
//! package) is unpacked and its `word/document.xml` is streamed for text runs.
//! Everything else reports a clear, actionable error instead of panicking.

use crate::{
    config::read_text_file,
    error::{AppError, Result},
};
use quick_xml::events::Event;
use std::path::Path;
use zip::ZipArchive;

/// Extract readable text from a TXT, PDF, or DOCX file based on its extension.
pub fn extract_text(path: &Path) -> Result<String> {
    let extension = path
        .extension()
        .and_then(|extension| extension.to_str())
        .unwrap_or_default()
        .to_lowercase();
    match extension.as_str() {
        "pdf" => extract_pdf(path),
        "docx" => extract_docx(path),
        "txt" | "md" | "" => read_text_file(path),
        other => Err(AppError::Message(format!(
            "Unsupported document type '.{other}'. Open a .txt, .pdf, or .docx file."
        ))),
    }
}

fn extract_pdf(path: &Path) -> Result<String> {
    pdf_extract::extract_text(path).map_err(|error| {
        AppError::Message(format!(
            "Could not read PDF '{}': {error}",
            path.display()
        ))
    })
}

fn extract_docx(path: &Path) -> Result<String> {
    let file = std::fs::File::open(path)?;
    let mut archive = ZipArchive::new(file).map_err(|error| {
        AppError::Message(format!(
            "Could not open DOCX '{}': {error}",
            path.display()
        ))
    })?;
    let mut entry = archive.by_name("word/document.xml").map_err(|_| {
        AppError::Message("The .docx file has no document body.".into())
    })?;
    let mut xml = String::new();
    std::io::Read::read_to_string(&mut entry, &mut xml)?;
    Ok(parse_document_xml(&xml))
}

/// Stream WordprocessingML, collecting `<w:t>` runs and emitting a newline at
/// every paragraph (`<w:p>`) and explicit break (`<w:br/>`) so paragraph
/// boundaries survive into synthesis.
fn parse_document_xml(xml: &str) -> String {
    let mut reader = quick_xml::Reader::from_str(xml);
    reader.config_mut().trim_text(false);
    let mut text = String::new();
    let mut in_run_text = false;
    loop {
        match reader.read_event() {
            Ok(Event::Start(element)) => match element.local_name().as_ref() {
                b"t" => in_run_text = true,
                b"tab" => text.push(' '),
                b"br" => text.push('\n'),
                _ => {}
            },
            Ok(Event::Empty(element)) => match element.local_name().as_ref() {
                b"br" => text.push('\n'),
                b"tab" => text.push(' '),
                _ => {}
            },
            Ok(Event::Text(content)) => {
                if in_run_text {
                    if let Ok(decoded) = content.unescape() {
                        text.push_str(&unescape(&decoded));
                    }
                }
            }
            Ok(Event::End(element)) => match element.local_name().as_ref() {
                b"t" => in_run_text = false,
                b"p" => text.push('\n'),
                _ => {}
            },
            Ok(Event::Eof) => break,
            Ok(_) => {}
            Err(error) => {
                eprintln!("Malformed DOCX XML, stopping parse: {error}");
                break;
            }
        }
    }
    collapse_blank_lines(&text)
}

/// Resolve the handful of XML entities that appear in Word text runs.
fn unescape(value: &str) -> String {
    value
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&apos;", "'")
        .replace("&amp;", "&")
}

/// Trim trailing whitespace per line and squash runs of blank lines so long
/// documents do not produce many silent paragraphs.
fn collapse_blank_lines(text: &str) -> String {
    let mut lines: Vec<String> = text
        .lines()
        .map(|line| line.trim_end().to_string())
        .collect();
    while lines.last().is_some_and(|line| line.is_empty()) {
        lines.pop();
    }
    let mut result = String::new();
    let mut previous_blank = false;
    for line in lines {
        let blank = line.trim().is_empty();
        if blank && previous_blank {
            continue;
        }
        if !result.is_empty() {
            result.push('\n');
        }
        result.push_str(&line);
        previous_blank = blank;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unsupported_extension_reports_error() {
        let path = Path::new("notes.rtf");
        let error = extract_text(path).unwrap_err().to_string();
        assert!(error.contains("Unsupported document type '.rtf'"));
    }

    #[test]
    fn missing_pdf_reports_error_not_panic() {
        let result = extract_text(Path::new("does-not-exist.pdf"));
        assert!(result.is_err());
    }

    #[test]
    fn document_xml_collects_runs_and_paragraphs() {
        let xml = "<w:body><w:p><w:r><w:t>Hello</w:t></w:r><w:r><w:t> world</w:t></w:r>\
                   <w:r><w:t>&amp; bye</w:t></w:r></w:p><w:p><w:t>Second &lt;line&gt;</w:t></w:p></w:body>";
        assert_eq!(
            parse_document_xml(xml),
            "Hello world& bye\nSecond <line>"
        );
    }

    #[test]
    fn docx_round_trip_from_in_memory_archive() {
        use std::io::Write;
        let directory = std::env::temp_dir();
        let path = directory.join(format!("hivetts-test-{}.docx", std::process::id()));
        {
            let file = std::fs::File::create(&path).expect("create zip");
            let mut builder = zip::ZipWriter::new(file);
            let options = zip::write::SimpleFileOptions::default()
                .compression_method(zip::CompressionMethod::Stored);
            builder
                .start_file("word/document.xml", options)
                .expect("start entry");
            builder
                .write_all(b"<w:document><w:body><w:p><w:t>Extracted text</w:t></w:p></w:body></w:document>")
                .expect("write entry");
            builder.finish().expect("finish zip");
        }
        let extracted = extract_text(&path).expect("docx extracts");
        let _ = std::fs::remove_file(&path);
        assert!(extracted.contains("Extracted text"));
    }
}
