//! Batch conversion: turn many documents into audio files in one pass.
//!
//! The TTS page collects a list of files or folders, extracts each document's
//! text, synthesizes it with the current voice settings, and encodes it into
//! the chosen container and quality. Failures are gathered per file so one
//! bad document never aborts the rest of the batch.

use crate::{
    config::AppConfig,
    documents,
    error::{AppError, Result},
    preprocess,
    tts::{ExportFormat, LocalSynthesizer, SpeechSynthesizer},
};
use std::{
    fs,
    path::{Path, PathBuf},
};

/// Outcome of a batch run: how many files converted and what failed.
#[derive(Debug)]
pub struct BatchReport {
    pub converted: usize,
    pub failures: Vec<(String, String)>,
}

/// Expand a user-entered list into concrete document paths. Entries are
/// separated by `;`, `,`, or newlines; a directory expands to the supported
/// documents directly inside it, sorted by name. Missing files are kept so
/// the report can name them instead of silently skipping them.
pub fn collect_inputs(spec: &str) -> Vec<PathBuf> {
    let mut inputs: Vec<PathBuf> = Vec::new();
    for entry in spec.split([';', ',', '\n']) {
        let entry = entry.trim().trim_matches('"').trim();
        if entry.is_empty() {
            continue;
        }
        let path = PathBuf::from(entry);
        if path.is_dir() {
            let mut found: Vec<PathBuf> = fs::read_dir(&path)
                .into_iter()
                .flatten()
                .filter_map(|entry| entry.ok().map(|entry| entry.path()))
                .filter(|path| path.is_file() && is_supported_document(path))
                .collect();
            found.sort();
            inputs.extend(found);
        } else {
            inputs.push(path);
        }
    }
    // Drop repeats (an explicit file may also appear in a scanned folder)
    // while keeping the user's order; Vec::dedup only folds adjacent runs.
    let mut seen = std::collections::HashSet::new();
    inputs.retain(|path| seen.insert(path.clone()));
    inputs
}

/// Document types the batch converter can read (same set as the Load button).
pub fn is_supported_document(path: &Path) -> bool {
    matches!(
        path.extension()
            .and_then(|extension| extension.to_str())
            .map(|extension| extension.to_lowercase())
            .as_deref(),
        Some("txt" | "md" | "pdf" | "docx")
    )
}

/// Convert every input into `format` inside `output_directory`, reporting
/// `(current, total, file name)` before each file so the UI can show progress.
pub fn convert_batch(
    inputs: &[PathBuf],
    format: ExportFormat,
    output_directory: &Path,
    config: &AppConfig,
    mut progress: impl FnMut(usize, usize, &str),
) -> Result<BatchReport> {
    if inputs.is_empty() {
        return Err(AppError::Message(
            "No input files to convert. List .txt, .md, .pdf, or .docx files, or a folder containing them."
                .into(),
        ));
    }
    fs::create_dir_all(output_directory)?;
    let synthesizer = LocalSynthesizer;
    let mut report = BatchReport {
        converted: 0,
        failures: Vec::new(),
    };
    for (index, input) in inputs.iter().enumerate() {
        let name = input
            .file_name()
            .map(|name| name.to_string_lossy().to_string())
            .unwrap_or_else(|| input.display().to_string());
        progress(index + 1, inputs.len(), &name);
        match convert_one(&synthesizer, input, format, output_directory, config) {
            Ok(()) => report.converted += 1,
            Err(error) => report.failures.push((name, error.to_string())),
        }
    }
    Ok(report)
}

fn convert_one(
    synthesizer: &LocalSynthesizer,
    input: &Path,
    format: ExportFormat,
    output_directory: &Path,
    config: &AppConfig,
) -> Result<()> {
    let extracted = documents::extract_text(input)?;
    let text = preprocess::normalize(&extracted);
    if text.trim().is_empty() {
        return Err(AppError::Message(
            "The document contains no readable text.".into(),
        ));
    }
    let stem = input
        .file_stem()
        .map(|stem| stem.to_string_lossy().to_string())
        .unwrap_or_else(|| "hivetts-batch".into());
    let destination = unique_destination(output_directory, &stem, format.extension());
    match format {
        ExportFormat::Wav => synthesizer.export_wav(&text, &destination, config),
        ExportFormat::Mp3 => synthesizer.export_mp3(&text, &destination, config),
        ExportFormat::Ogg => synthesizer.export_ogg(&text, &destination, config),
    }
}

/// Keep the source document's stem; add a counter instead of overwriting an
/// existing export with the same name.
fn unique_destination(directory: &Path, stem: &str, extension: &str) -> PathBuf {
    let mut candidate = directory.join(format!("{stem}.{extension}"));
    let mut counter = 1u32;
    while candidate.exists() {
        candidate = directory.join(format!("{stem}-{counter}.{extension}"));
        counter += 1;
    }
    candidate
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn collect_inputs_splits_trims_and_expands_directories() {
        let directory = std::env::temp_dir().join(format!(
            "hivetts-batch-test-{}",
            std::process::id()
        ));
        let _ = fs::create_dir_all(&directory);
        fs::write(directory.join("b.txt"), "second").expect("write b");
        fs::write(directory.join("a.md"), "first").expect("write a");
        fs::write(directory.join("skip.bin"), "binary").expect("write bin");
        let spec = format!(
            "\"{}\" ; {} , {}\n{}",
            directory.join("a.md").display(),
            directory.join("b.txt").display(),
            directory.join("missing.txt").display(),
            directory.display(),
        );
        let inputs = collect_inputs(&spec);
        // a.md and b.txt appear twice (explicit + folder scan) but dedup to
        // one each; the folder scan adds them in sorted order; missing stays.
        assert_eq!(inputs.len(), 3);
        assert!(inputs.contains(&directory.join("a.md")));
        assert!(inputs.contains(&directory.join("b.txt")));
        assert!(inputs.contains(&directory.join("missing.txt")));
        assert!(!inputs.contains(&directory.join("skip.bin")));
        assert!(collect_inputs("  ; , \n ").is_empty());
        let _ = fs::remove_dir_all(&directory);
    }

    #[test]
    fn supported_documents_match_the_load_button() {
        assert!(is_supported_document(Path::new("book.TXT")));
        assert!(is_supported_document(Path::new("notes.md")));
        assert!(is_supported_document(Path::new("paper.pdf")));
        assert!(is_supported_document(Path::new("letter.docx")));
        assert!(!is_supported_document(Path::new("audio.wav")));
        assert!(!is_supported_document(Path::new("README")));
    }

    #[test]
    fn unique_destination_never_overwrites() {
        let directory = std::env::temp_dir().join(format!(
            "hivetts-batch-unique-{}",
            std::process::id()
        ));
        let _ = fs::create_dir_all(&directory);
        let first = unique_destination(&directory, "chapter", "mp3");
        assert_eq!(first, directory.join("chapter.mp3"));
        fs::write(&first, b"existing").expect("write existing");
        let second = unique_destination(&directory, "chapter", "mp3");
        assert_eq!(second, directory.join("chapter-1.mp3"));
        let _ = fs::remove_dir_all(&directory);
    }

    #[test]
    fn empty_batch_reports_a_clear_error() {
        let config = AppConfig::default();
        let error = convert_batch(
            &[],
            ExportFormat::Mp3,
            Path::new("/tmp/hivetts-unused"),
            &config,
            |_, _, _| {},
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("No input files"));
    }

    #[test]
    #[ignore = "needs espeak-ng and ffmpeg installed"]
    fn batch_and_exports_with_real_engines() {
        let directory = std::env::temp_dir().join(format!(
            "hivetts-batch-e2e-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&directory);
        fs::create_dir_all(&directory).expect("create e2e dir");
        fs::write(directory.join("one.txt"), "Hello from the first document.")
            .expect("write one");
        fs::write(directory.join("two.txt"), "And hello from the second document.")
            .expect("write two");
        let out = directory.join("out");
        let config = AppConfig {
            tts_language: "en".into(),
            output_quality: "standard".into(),
            auto_save_directory: out.display().to_string(),
            ..AppConfig::default()
        };
        let inputs = collect_inputs(&directory.display().to_string());
        assert_eq!(inputs.len(), 2);
        let mut progress_calls = 0;
        let report =
            convert_batch(&inputs, ExportFormat::Mp3, &out, &config, |_, _, _| {
                progress_calls += 1;
            })
            .expect("batch conversion");
        assert_eq!(report.converted, 2);
        assert!(report.failures.is_empty());
        assert_eq!(progress_calls, 2);
        assert!(out.join("one.mp3").is_file());
        assert!(out.join("two.mp3").is_file());

        // OGG export and a WAV export resampled to the High preset's rate.
        let synthesizer = LocalSynthesizer;
        let ogg = out.join("voice.ogg");
        synthesizer
            .export_ogg("Testing the OGG encoder.", &ogg, &config)
            .expect("ogg export");
        assert!(ogg.is_file());
        let wav = out.join("voice.wav");
        let high = AppConfig {
            output_quality: "high".into(),
            ..config.clone()
        };
        synthesizer
            .export_wav("Testing the WAV resampler.", &wav, &high)
            .expect("wav export");
        assert!(wav.is_file());
        assert!(fs::metadata(&wav).expect("wav metadata").len() > 44);
        let _ = fs::remove_dir_all(&directory);
    }
}
