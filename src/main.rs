slint::include_modules!();

use serde::Deserialize;
use slint::{Model, SharedString};
use std::{
    env,
    io::{self, Write},
    path::PathBuf,
    process::Command,
    sync::{
        atomic::{AtomicI32, AtomicU64, Ordering},
        Arc,
    },
    thread,
    time::Duration,
};
use tts::{
    audio,
    config::{self, AppConfig},
    stt::{SpeechRecognizer, WhisperCliRecognizer},
    tts::{LocalSynthesizer, SpeechSynthesizer},
};

static MODEL_SEARCH_GENERATION: AtomicU64 = AtomicU64::new(0);
// Indexes already applied to the dynamic TTS voice and language selectors,
// so list refreshes do not re-trigger the selection handlers.
static TTS_LANGUAGE_APPLIED: AtomicI32 = AtomicI32::new(-1);
static TTS_VOICE_APPLIED: AtomicI32 = AtomicI32::new(-1);
static TTS_VOICE_ROW_APPLIED: AtomicI32 = AtomicI32::new(-1);
static STT_MODEL_APPLIED: AtomicI32 = AtomicI32::new(-1);

#[derive(Clone, Deserialize)]
struct LanguageOptionData {
    name: String,
    code: String,
}

fn set_language_options(app: &AppWindow, mut options: Vec<LanguageOption>) {
    options.insert(
        0,
        LanguageOption {
            name: "All languages".into(),
            code: "".into(),
        },
    );
    let labels = options
        .iter()
        .map(|language| {
            if language.code.is_empty() {
                language.name.clone()
            } else {
                format!("{} ({})", language.name, language.code).into()
            }
        })
        .collect::<Vec<_>>();
    app.set_language_options(slint::ModelRc::new(slint::VecModel::from(options)));
    app.set_language_option_labels(slint::ModelRc::new(slint::VecModel::from(labels)));
}

/// Language name shown above the voice rows that belong to it in the
/// categorized list on the TTS page.
fn voice_group_name(language: &str, catalog: &[LanguageOptionData]) -> String {
    if language == "other" {
        return "Unspecified".into();
    }
    let code = language.to_uppercase();
    match catalog.iter().find(|entry| entry.code == language) {
        Some(entry) => format!("{} ({code})", entry.name),
        None => code,
    }
}

/// Downloaded Piper voices and the built-in engine as (language code, id,
/// model file) triples in display order.
type VoiceEntry = (String, String, String);

/// Build the categorized voice-model rows and the parallel entry arrays used
/// when a row is selected. Each downloaded Piper voice becomes a selectable
/// row placed under its language header, alongside the built-in engine and an
/// optional custom-path row.
fn build_voice_rows(
    installed: &[tts::models::HubModel],
    piper_voice: &str,
    catalog: &[LanguageOptionData],
) -> (Vec<VoiceRow>, Vec<VoiceEntry>) {
    let mut downloaded: Vec<VoiceEntry> = Vec::new();
    for model in installed {
        if let Some(file) = tts::models::tts_voice_file(model) {
            let language = tts::models::parse_languages(&model.languages)
                .into_iter()
                .next()
                .unwrap_or_else(|| "other".into());
            downloaded.push((language, model.id.clone(), file));
        }
    }
    downloaded.sort();

    // Entry 0 is always the built-in engine, which needs no model file.
    let mut entries: Vec<VoiceEntry> = vec![("".into(), "".into(), "".into())];
    // A manually configured Piper path that no downloaded model provides stays
    // selectable instead of silently switching back to the built-in engine.
    if !piper_voice.is_empty() && !downloaded.iter().any(|(_, _, file)| file == piper_voice) {
        entries.push(("other".into(), "custom".into(), piper_voice.to_owned()));
    }
    entries.extend(downloaded.iter().cloned());

    let header = |text: String| VoiceRow {
        is_header: true,
        text: text.into(),
        entry_index: -1,
        selected: false,
    };
    let option = |text: String, entry_index: usize| VoiceRow {
        is_header: false,
        text: text.into(),
        entry_index: entry_index as i32,
        selected: entries[entry_index].2 == piper_voice,
    };
    let mut rows: Vec<VoiceRow> = vec![
        header("Engines".into()),
        option("Built-in engine (eSpeak NG)".into(), 0),
    ];
    if entries.len() > 1 && entries[1].1 == "custom" {
        rows.push(header("Custom".into()));
        rows.push(option("Custom Piper voice".into(), 1));
    }
    let mut current_group = String::new();
    for (index, (language, id, _)) in downloaded.iter().enumerate() {
        // Downloaded voices follow the built-in (and optional custom) entries.
        let entry_index = index + entries.len() - downloaded.len();
        if language != &current_group {
            current_group = language.clone();
            rows.push(header(voice_group_name(language, catalog)));
        }
        rows.push(option(id.clone(), entry_index));
    }
    (rows, entries)
}

/// Move the voice selection to a categorized-list row and apply its model.
fn select_voice_row(app: &AppWindow, weak: &slint::Weak<AppWindow>, row: i32) {
    let Some(row_data) = app.get_tts_voice_rows().row_data(row.max(0) as usize) else {
        return;
    };
    if row_data.is_header {
        return;
    }
    let index = row_data.entry_index.max(0) as usize;
    TTS_VOICE_ROW_APPLIED.store(row, Ordering::SeqCst);
    app.set_tts_voice_row_selected_index(row);
    // Store the applied entry so the changed tts-voice-index handler does not
    // re-apply the same selection.
    TTS_VOICE_APPLIED.store(index as i32, Ordering::SeqCst);
    app.set_tts_voice_index(index as i32);
    let Some(id) = app.get_tts_voice_model_ids().row_data(index) else {
        return;
    };
    let id = id.to_string();
    if id.is_empty() {
        // The built-in eSpeak NG engine needs no model files.
        app.set_piper_voice("".into());
        set_status(weak.clone(), "Switched to the built-in engine.".into());
        return;
    }
    if id == "custom" {
        set_status(weak.clone(), "Using the custom Piper voice path.".into());
        return;
    }
    let Some(file) = app.get_tts_voice_model_files().row_data(index) else {
        return;
    };
    app.set_piper_voice(file.to_string().into());
    if let Some(language) = app.get_tts_voice_languages().row_data(index) {
        let language = language.to_string();
        if !language.is_empty() && language != "other" {
            set_selected_language(app, &language);
        }
    }
    set_status(weak.clone(), format!("Voice model set to {id}."));
}

/// Rebuild the categorized voice-model list and the language selector on the
/// TTS page from the downloaded models, grouped by their declared languages.
fn apply_tts_voices(weak: &slint::Weak<AppWindow>, catalog: &[LanguageOptionData]) {
    let Some(app) = weak.upgrade() else {
        return;
    };
    let installed = tts::models::installed_models().unwrap_or_default();
    let piper_voice = app.get_piper_voice().to_string();
    let (rows, entries) = build_voice_rows(&installed, &piper_voice, catalog);
    let selected_row = rows
        .iter()
        .position(|row| row.selected)
        .unwrap_or(0) as i32;
    let selected_entry = rows
        .iter()
        .find(|row| !row.is_header && row.selected)
        .map(|row| row.entry_index)
        .unwrap_or(0);
    TTS_VOICE_ROW_APPLIED.store(selected_row, Ordering::SeqCst);

    let labels: Vec<SharedString> = entries
        .iter()
        .map(|(language, id, _)| {
            if id.is_empty() {
                "Built-in engine (eSpeak NG)".into()
            } else if id == "custom" {
                "Custom Piper voice".into()
            } else {
                format!("{} · {}", language_display(language, catalog), id).into()
            }
        })
        .collect();
    let ids: Vec<SharedString> = entries
        .iter()
        .map(|(_, id, _)| id.clone().into())
        .collect();
    let files: Vec<SharedString> = entries
        .iter()
        .map(|(_, _, file)| file.clone().into())
        .collect();
    let voice_languages: Vec<SharedString> = entries
        .iter()
        .map(|(language, _, _)| language.clone().into())
        .collect();

    // The supported built-in languages stay on top; model languages extend the list.
    let mut language_codes: Vec<String> =
        ["fa", "en", "ar", "tr"].iter().map(|code| code.to_string()).collect();
    for (language, _, _) in &entries {
        if language != "other" && !language_codes.contains(language) {
            language_codes.push(language.clone());
        }
    }
    let current_language = app.get_tts_language().to_string();
    if !current_language.is_empty() && !language_codes.contains(&current_language) {
        language_codes.push(current_language.clone());
    }
    let language_labels = language_codes
        .iter()
        .map(|code| language_display(code, catalog).into())
        .collect::<Vec<SharedString>>();
    let language_index = language_codes
        .iter()
        .position(|code| *code == current_language)
        .unwrap_or(0) as i32;

    let summary = tts::models::group_by_language(&installed)
        .iter()
        .map(|group| {
            format!(
                "{} · {}",
                language_display(&group.language, catalog),
                group.models.len()
            )
        })
        .collect::<Vec<_>>()
        .join("  |  ");

    app.set_tts_voice_rows(slint::ModelRc::new(slint::VecModel::from(rows)));
    app.set_tts_voice_row_selected_index(selected_row);
    app.set_tts_voice_labels(slint::ModelRc::new(slint::VecModel::from(labels)));
    app.set_tts_voice_model_ids(slint::ModelRc::new(slint::VecModel::from(ids)));
    app.set_tts_voice_model_files(slint::ModelRc::new(slint::VecModel::from(files)));
    app.set_tts_voice_languages(slint::ModelRc::new(slint::VecModel::from(voice_languages)));
    app.set_tts_language_labels(slint::ModelRc::new(slint::VecModel::from(language_labels)));
    app.set_tts_language_codes(slint::ModelRc::new(slint::VecModel::from(
        language_codes
            .iter()
            .map(|code| SharedString::from(code.clone()))
            .collect::<Vec<_>>(),
    )));
    app.set_installed_language_summary(summary.into());
    // Store the applied indexes first so the setter calls below do not
    // re-enter the selection handlers.
    TTS_VOICE_APPLIED.store(selected_entry, Ordering::SeqCst);
    app.set_tts_voice_index(selected_entry);
    TTS_LANGUAGE_APPLIED.store(language_index, Ordering::SeqCst);
    app.set_tts_language_index(language_index);
    app.set_tts_language(current_language.into());
}

/// Rebuild the categorized STT model selector on the Speech-to-Text page from
/// the downloaded automatic-speech-recognition models, grouped by language.
fn apply_stt_models(weak: &slint::Weak<AppWindow>, catalog: &[LanguageOptionData]) {
    let Some(app) = weak.upgrade() else {
        return;
    };
    let installed = tts::models::installed_models().unwrap_or_default();
    let mut entries: Vec<(String, String, String)> = Vec::new();
    for model in &installed {
        let Some(file) = tts::models::stt_model_file(model) else {
            continue;
        };
        let language = tts::models::parse_languages(&model.languages)
            .into_iter()
            .next()
            .unwrap_or_else(|| "other".into());
        entries.push((language, model.id.clone(), file));
    }
    entries.sort();

    let current_path = app.get_model_path().to_string();
    let mut labels: Vec<SharedString> = Vec::new();
    let mut paths: Vec<SharedString> = Vec::new();
    let mut model_index = 0i32;
    for (language, id, file) in &entries {
        labels.push(format!("{} · {}", language_display(language, catalog), id).into());
        paths.push(file.clone().into());
        if file == &current_path {
            model_index = (labels.len() - 1) as i32;
        }
    }
    // A manually configured path that no downloaded model provides stays as the
    // default selection so the selector does not overwrite it.
    if !current_path.is_empty() && !entries.iter().any(|(_, _, file)| file == &current_path) {
        labels.push("Custom model path".into());
        paths.push(current_path.clone().into());
        model_index = (labels.len() - 1) as i32;
    }

    app.set_stt_model_labels(slint::ModelRc::new(slint::VecModel::from(labels)));
    app.set_stt_model_paths(slint::ModelRc::new(slint::VecModel::from(paths)));
    STT_MODEL_APPLIED.store(model_index, Ordering::SeqCst);
    app.set_stt_model_index(model_index);
}

/// Select the language row matching `code` without re-running the handler.
fn set_selected_language(app: &AppWindow, code: &str) {
    let codes = app.get_tts_language_codes();
    if let Some(row) = (0..codes.row_count())
        .find(|row| codes.row_data(*row).is_some_and(|value| value.as_str() == code))
    {
        TTS_LANGUAGE_APPLIED.store(row as i32, Ordering::SeqCst);
        app.set_tts_language_index(row as i32);
    }
    app.set_tts_language(code.into());
}

fn language_display(code: &str, catalog: &[LanguageOptionData]) -> String {
    if code == "other" {
        return "Unspecified".into();
    }
    if let Some(entry) = catalog.iter().find(|entry| entry.code == code) {
        return format!("{} ({code})", entry.name);
    }
    let mut characters = code.chars();
    match characters.next() {
        Some(first) => first.to_uppercase().collect::<String>() + characters.as_str(),
        None => String::new(),
    }
}

/// Queue a voice-list rebuild on the UI thread after the installed models change.
fn schedule_tts_voice_refresh(weak: slint::Weak<AppWindow>, catalog: Arc<Vec<LanguageOptionData>>) {
    if let Err(error) = slint::invoke_from_event_loop(move || {
        if let Some(app) = weak.upgrade() {
            apply_tts_voices(&app.as_weak(), &catalog);
            apply_stt_models(&app.as_weak(), &catalog);
        }
    }) {
        eprintln!("Could not refresh the model lists: {error}");
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let app = AppWindow::new()?;
    // Language names and ISO codes are from iso-codes (LGPL-2.1-or-later).
    let language_catalog = Arc::new(
        serde_json::from_str::<Vec<LanguageOptionData>>(include_str!("../ui/languages.json"))?,
    );
    let language_options = language_catalog
        .iter()
        .take(100)
        .map(|language| LanguageOption {
            name: language.name.clone().into(),
            code: language.code.clone().into(),
        })
        .collect::<Vec<_>>();
    set_language_options(&app, language_options);
    let weak = app.as_weak();
    let catalog = Arc::clone(&language_catalog);
    app.on_search_languages(move |query| {
        let query = query.trim().to_lowercase();
        let options = catalog
            .iter()
            .filter(|language| {
                query.is_empty()
                    || language.name.to_lowercase().contains(&query)
                    || language.code.to_lowercase().starts_with(&query)
            })
            .take(100)
            .map(|language| LanguageOption {
                name: language.name.clone().into(),
                code: language.code.clone().into(),
            })
            .collect::<Vec<_>>();
        if let Some(app) = weak.upgrade() {
            set_language_options(&app, options);
        }
    });
    let mut settings = match AppConfig::load() {
        Ok(settings) => settings,
        Err(error) => {
            eprintln!("Could not load settings: {error}");
            AppConfig::default()
        }
    };

    if let Ok(model_path) = env::var("WHISPER_MODEL") {
        settings.whisper_model = model_path;
    }
    app.set_model_path(settings.whisper_model.clone().into());
    app.set_piper_executable(settings.piper_executable.clone().into());
    app.set_piper_voice(settings.piper_voice.clone().into());
    app.set_language(settings.language.clone().into());
    app.set_speech_rate(settings.speech_rate);
    app.set_theme(settings.theme.clone().into());
    app.set_voice(settings.voice.clone().into());
    app.set_tts_language(settings.tts_language.clone().into());
    app.set_voice_style(settings.voice_style.clone().into());
    app.set_speech_tone(settings.speech_tone.clone().into());
    app.set_pitch(settings.pitch);
    app.set_volume(settings.volume);
    app.set_pause_sentence(settings.pause_sentence);
    app.set_pause_paragraph(settings.pause_paragraph);
    app.set_auto_save_directory(settings.auto_save_directory.clone().into());

    // Build the dynamic voice and language lists from downloaded models.
    apply_tts_voices(&app.as_weak(), &language_catalog);
    apply_stt_models(&app.as_weak(), &language_catalog);

    let weak = app.as_weak();
    let catalog = Arc::clone(&language_catalog);
    app.on_refresh_tts_voices(move || apply_tts_voices(&weak, &catalog));

    let weak = app.as_weak();
    let catalog = Arc::clone(&language_catalog);
    app.on_refresh_stt_models(move || apply_stt_models(&weak, &catalog));

    let weak = app.as_weak();
    app.on_language_changed(move |index| {
        if TTS_LANGUAGE_APPLIED.load(Ordering::SeqCst) == index {
            return;
        }
        let Some(app) = weak.upgrade() else {
            return;
        };
        let Some(code) = app
            .get_tts_language_codes()
            .row_data(index.max(0) as usize)
        else {
            return;
        };
        TTS_LANGUAGE_APPLIED.store(index, Ordering::SeqCst);
        app.set_tts_language(code.to_string().into());
    });

    let weak = app.as_weak();
    app.on_voice_changed(move |index| {
        if TTS_VOICE_APPLIED.load(Ordering::SeqCst) == index {
            return;
        }
        let Some(app) = weak.upgrade() else {
            return;
        };
        // Mirror the entry selection onto the matching categorized-list row.
        let row = app
            .get_tts_voice_rows()
            .iter()
            .position(|row| !row.is_header && row.entry_index == index)
            .unwrap_or(0) as i32;
        // The selection may already be applied by a categorized-list click.
        if TTS_VOICE_ROW_APPLIED.load(Ordering::SeqCst) == row {
            return;
        }
        select_voice_row(&app, &weak.clone(), row);
    });

    let weak = app.as_weak();
    app.on_voice_picked(move |row| {
        let Some(app) = weak.upgrade() else {
            return;
        };
        if TTS_VOICE_ROW_APPLIED.load(Ordering::SeqCst) == row {
            return;
        }
        select_voice_row(&app, &weak.clone(), row);
    });

    let recording = audio::new_recording_state();

    let weak = app.as_weak();
    app.on_update_text_stats(move |text| {
        if let Some(app) = weak.upgrade() {
            app.set_text_stats(
                format!(
                    "{} characters · {} words",
                    text.chars().count(),
                    text.split_whitespace().count()
                )
                .into(),
            );
        }
    });

    let weak = app.as_weak();
    app.on_speak(move |text| {
        let weak = weak.clone();
        let text = text.to_string();
        thread::spawn(move || {
            let settings = load_runtime_config(&weak);
            let result = LocalSynthesizer.speak(&text, &settings);
            let message = match result {
                Ok(Some(path)) => {
                    let message = format!("Speech finished. File saved to {}.", path.display());
                    set_last_generated_file(weak.clone(), path.display().to_string());
                    message
                }
                Ok(None) => "Speech finished.".to_string(),
                Err(error) => error.to_string(),
            };
            set_status(weak, message);
        });
    });

    let weak = app.as_weak();
    app.on_download_model(move |size| {
        let weak = weak.clone();
        let size = match size.as_str() {
            "tiny" => tts::models::WhisperSize::Tiny,
            "base" => tts::models::WhisperSize::Base,
            "small" => tts::models::WhisperSize::Small,
            _ => {
                set_status(weak, "Unknown Whisper model size.".into());
                return;
            }
        };
        set_status(weak.clone(), "Downloading speech model…".into());
        thread::spawn(move || match tts::models::download_model(size) {
            Ok(path) => {
                let path_for_status = path.display().to_string();
                if let Err(error) = slint::invoke_from_event_loop(move || {
                    if let Some(app) = weak.upgrade() {
                        app.set_model_path(path.to_string_lossy().to_string().into());
                        app.set_status_text(format!("Model downloaded: {path_for_status}").into());
                    }
                }) {
                    eprintln!("Could not update the UI: {error}");
                }
            }
            Err(error) => set_status(weak, format!("Model download failed: {error}")),
        });
    });

    let weak = app.as_weak();
    app.on_refresh_models(
        move |query, category, max_size, language, framework, hardware| {
            load_model_cards(
                weak.clone(),
                query.to_string(),
                category.to_string(),
                max_size.to_string(),
                language.to_string(),
                framework.to_string(),
                hardware.to_string(),
            );
        },
    );

    let weak = app.as_weak();
    let catalog = Arc::clone(&language_catalog);
    app.on_download_hub_model(move |id, task| {
        let weak = weak.clone();
        let refresh_weak = weak.clone();
        let refresh_catalog = Arc::clone(&catalog);
        let id = id.to_string();
        let task = task.to_string();
        set_active_model_download(weak.clone(), id.clone());
        set_model_download_progress_text(weak.clone(), "Preparing download…".into());
        set_model_download_progress(weak.clone(), 0.0);
        set_model_download_progress_indeterminate(weak.clone(), true);
        set_status(weak.clone(), format!("Preparing download for {id}…"));
        thread::spawn(move || {
            let progress_weak = weak.clone();
            let result = tts::models::download_hub_model(
                &id,
                &task,
                move |downloaded, total, file, speed| {
                    let percentage = if total == 0 {
                        0.0
                    } else {
                        (downloaded as f64 / total as f64).clamp(0.0, 1.0) as f32
                    };
                    let remaining = if speed > 0.0 && total > downloaded {
                        format!(
                            " · {} left",
                            format_duration(((total - downloaded) as f64 / speed) as u64)
                        )
                    } else {
                        String::new()
                    };
                    let progress_label = if total == 0 {
                        "Downloading".to_owned()
                    } else {
                        format!("{}%", (percentage * 100.0).round() as u32)
                    };
                    let text = format!(
                        "{} · {} / {} · {}/s{} · {}",
                        progress_label,
                        tts::models::format_bytes(downloaded),
                        if total == 0 {
                            "unknown size".into()
                        } else {
                            tts::models::format_bytes(total)
                        },
                        tts::models::format_bytes(speed.max(0.0) as u64),
                        remaining,
                        file,
                    );
                    let progress_tick_weak = progress_weak.clone();
                    if let Err(error) = slint::invoke_from_event_loop(move || {
                        if let Some(app) = progress_tick_weak.upgrade() {
                            app.set_model_download_progress(percentage);
                            app.set_model_download_progress_indeterminate(total == 0);
                            app.set_model_download_progress_text(text.into());
                        }
                    }) {
                        eprintln!("Could not update model download progress: {error}");
                    }
                },
            );
            match result {
                Ok(path) => {
                    set_active_model_download(weak.clone(), String::new());
                    set_model_download_progress(weak.clone(), 1.0);
                    set_model_download_progress_indeterminate(weak.clone(), false);
                    set_status(
                        weak.clone(),
                        format!("Model installed and verified at {}.", path.display()),
                    );
                    set_model_download_progress_text(
                        weak.clone(),
                        format!("Installed: {}", path.display()),
                    );
                    load_model_cards(
                        weak,
                        String::new(),
                        "installed".into(),
                        String::new(),
                        String::new(),
                        String::new(),
                        String::new(),
                    );
                    schedule_tts_voice_refresh(refresh_weak, refresh_catalog);
                }
                Err(error) => {
                    set_active_model_download(weak.clone(), String::new());
                    set_model_download_progress_indeterminate(weak.clone(), false);
                    let message = error.to_string();
                    set_model_download_progress_text(
                        weak.clone(),
                        if message.starts_with("Download cancelled") {
                            "Download cancelled; select Download again to resume.".into()
                        } else {
                            String::new()
                        },
                    );
                    set_status(weak, format!("Model download failed: {error}"));
                }
            }
        });
    });

    let weak = app.as_weak();
    app.on_cancel_model_download(move |model| {
        match tts::models::cancel_hub_download(model.as_str()) {
            Ok(()) => set_status(weak.clone(), "Cancelling model download…".into()),
            Err(error) => set_status(weak.clone(), format!("Could not cancel download: {error}")),
        }
    });

    let weak = app.as_weak();
    app.on_open_model_folder(move |path| {
        let path = PathBuf::from(path.to_string());
        if let Err(error) = tts::models::open_model_folder(&path) {
            set_status(
                weak.clone(),
                format!("Could not open model folder: {error}"),
            );
        }
    });

    let weak = app.as_weak();
    app.on_open_hub_repository(move |model| {
        let id = model.to_string();
        if !id.split('/').all(|part| {
            !part.is_empty()
                && part.chars().all(|character| {
                    character.is_ascii_alphanumeric() || "-_. ".contains(character)
                })
        }) {
            set_status(weak.clone(), "Invalid Hugging Face repository ID.".into());
            return;
        }
        let url = format!("https://huggingface.co/{id}");
        let result = open_url(&url);
        if let Err(error) = result {
            set_status(weak.clone(), format!("Could not open repository: {error}"));
        }
    });

    let weak = app.as_weak();
    let catalog = Arc::clone(&language_catalog);
    app.on_delete_model(move |model| {
        let weak = weak.clone();
        let refresh_weak = weak.clone();
        let refresh_catalog = Arc::clone(&catalog);
        let id = model.to_string();
        thread::spawn(move || match tts::models::delete_hub_model(&id) {
            Ok(()) => {
                set_status(weak.clone(), format!("Deleted installed model {id}."));
                load_model_cards(
                    weak,
                    String::new(),
                    "installed".into(),
                    String::new(),
                    String::new(),
                    String::new(),
                    String::new(),
                );
                schedule_tts_voice_refresh(refresh_weak, refresh_catalog);
            }
            Err(error) => set_status(weak, format!("Could not delete model: {error}")),
        });
    });

    let weak = app.as_weak();
    app.on_load_text(move |path| {
        let weak = weak.clone();
        let path = PathBuf::from(path.to_string());
        thread::spawn(move || match config::read_text_file(&path) {
            Ok(text) => {
                if let Err(error) = slint::invoke_from_event_loop(move || {
                    if let Some(app) = weak.upgrade() {
                        app.set_speech_text(text.into());
                        app.set_status_text("Text file loaded.".into());
                    }
                }) {
                    eprintln!("Could not update the UI: {error}");
                }
            }
            Err(error) => set_status(weak, format!("Could not load text: {error}")),
        });
    });

    let weak = app.as_weak();
    app.on_export_wav(move |destination| {
        let weak = weak.clone();
        let destination = PathBuf::from(destination.to_string());
        let Some(app) = weak.upgrade() else {
            return;
        };
        let text = app.get_speech_text().to_string();
        drop(app);
        thread::spawn(move || {
            let settings = load_runtime_config(&weak);
            let result = LocalSynthesizer.export_wav(&text, &destination, &settings);
            set_status(
                weak,
                result
                    .map(|()| format!("WAV exported to {}.", destination.display()))
                    .unwrap_or_else(|error| error.to_string()),
            );
        });
    });

    let weak = app.as_weak();
    app.on_export_mp3(move |destination| {
        let weak = weak.clone();
        let destination = PathBuf::from(destination.to_string());
        let Some(app) = weak.upgrade() else {
            return;
        };
        let text = app.get_speech_text().to_string();
        drop(app);
        thread::spawn(move || {
            let settings = load_runtime_config(&weak);
            let result = LocalSynthesizer.export_mp3(&text, &destination, &settings);
            set_status(
                weak,
                result
                    .map(|()| format!("MP3 exported to {}.", destination.display()))
                    .unwrap_or_else(|error| error.to_string()),
            );
        });
    });

    let weak = app.as_weak();
    app.on_open_export_folder(move |path| {
        let path = PathBuf::from(path.to_string());
        if let Err(error) = tts::tts::open_path(&path) {
            set_status(
                weak.clone(),
                format!("Could not open the export folder: {error}"),
            );
        }
    });

    let weak = app.as_weak();
    app.on_transcribe_file(move |audio_path| {
        let weak = weak.clone();
        let audio_path = PathBuf::from(audio_path.to_string());
        let settings = load_runtime_config(&weak);
        let model_path = PathBuf::from(settings.whisper_model.clone());
        set_status(weak.clone(), "Transcribing audio file…".into());
        thread::spawn(move || {
            let result =
                WhisperCliRecognizer.transcribe(&model_path, &audio_path, &settings.language);
            match result {
                Ok(text) => {
                    if let Err(error) = slint::invoke_from_event_loop(move || {
                        if let Some(app) = weak.upgrade() {
                            app.set_transcript_text(text.into());
                            app.set_status_text("Transcription finished.".into());
                        }
                    }) {
                        eprintln!("Could not update the UI: {error}");
                    }
                }
                Err(error) => set_status(weak, error.to_string()),
            }
        });
    });

    let weak = app.as_weak();
    app.on_save_settings(move || {
        let Some(app) = weak.upgrade() else {
            return;
        };
        let settings = runtime_config(&app);
        drop(app);
        match settings.save() {
            Ok(()) => {}
            Err(error) => set_status(weak.clone(), format!("Could not save settings: {error}")),
        }
    });

    let weak = app.as_weak();
    let active_recording = recording.clone();
    app.on_start_recording(move || {
        let mut active = match active_recording.lock() {
            Ok(active) => active,
            Err(error) => {
                set_status(
                    weak.clone(),
                    format!("Recording state is unavailable: {error}"),
                );
                return;
            }
        };
        if active.is_some() {
            set_status(weak.clone(), "A recording is already in progress.".into());
            return;
        }

        let path = audio::recording_path();
        match audio::start_recording(&path) {
            Ok(child) => {
                *active = Some((child, path));
                if let Some(app) = weak.upgrade() {
                    app.set_is_recording(true);
                    app.set_status_text("Recording…".into());
                }
                let weak = weak.clone();
                audio::monitor_recording(active_recording.clone(), move |failure| {
                    if let Err(error) = slint::invoke_from_event_loop(move || {
                        if let Some(app) = weak.upgrade() {
                            app.set_is_recording(false);
                            app.set_status_text(failure.into());
                        }
                    }) {
                        eprintln!("Could not update the UI: {error}");
                    }
                });
            }
            Err(error) => {
                if let Some(app) = weak.upgrade() {
                    app.set_status_text(format!("Could not start recording: {error}").into());
                }
            }
        }
    });

    let weak = app.as_weak();
    let active_recording = recording.clone();
    app.on_stop_recording(move || {
        let recording = match active_recording.lock() {
            Ok(mut active) => active.take(),
            Err(error) => {
                set_status(
                    weak.clone(),
                    format!("Recording state is unavailable: {error}"),
                );
                return;
            }
        };
        let Some((mut child, path)) = recording else {
            if let Some(app) = weak.upgrade() {
                app.set_is_recording(false);
                app.set_status_text("There is no active recording.".into());
            }
            return;
        };

        if let Some(app) = weak.upgrade() {
            app.set_is_recording(false);
            app.set_status_text("Finishing recording…".into());
        }

        let weak = weak.clone();
        thread::spawn(move || {
            let stop_result = child
                .stdin
                .as_mut()
                .ok_or_else(|| io::Error::other("recording process stdin is unavailable"))
                .and_then(|stdin| stdin.write_all(b"q\n"));

            if let Err(error) = stop_result {
                if let Err(cleanup_error) = child.kill() {
                    eprintln!("Could not terminate recording process: {cleanup_error}");
                }
                if let Err(cleanup_error) = child.wait() {
                    eprintln!("Could not reap recording process: {cleanup_error}");
                }
                audio::remove_recording(&path);
                set_status(weak, format!("Could not stop recording: {error}"));
                return;
            }

            match child.wait() {
                Ok(status) if status.success() => {
                    transcribe_recording(weak, path);
                }
                Ok(status) => {
                    let diagnostic = child
                        .stderr
                        .take()
                        .map(audio::read_stderr)
                        .unwrap_or_else(|| "No diagnostic output available.".into());
                    audio::remove_recording(&path);
                    set_status(weak, format!("Recording failed ({status}): {diagnostic}"));
                }
                Err(error) => {
                    audio::remove_recording(&path);
                    set_status(weak, format!("Could not finish recording: {error}"));
                }
            }
        });
    });

    load_model_cards(
        app.as_weak(),
        String::new(),
        "all".into(),
        String::new(),
        String::new(),
        String::new(),
        String::new(),
    );
    app.run()?;
    Ok(())
}

fn runtime_config(app: &AppWindow) -> AppConfig {
    AppConfig {
        whisper_model: app.get_model_path().to_string(),
        piper_executable: app.get_piper_executable().to_string(),
        piper_voice: app.get_piper_voice().to_string(),
        speech_rate: app.get_speech_rate(),
        language: app.get_language().to_string(),
        tts_language: app.get_tts_language().to_string(),
        theme: app.get_theme().to_string(),
        voice: app.get_voice().to_string(),
        pitch: app.get_pitch(),
        volume: app.get_volume(),
        voice_style: app.get_voice_style().to_string(),
        speech_tone: app.get_speech_tone().to_string(),
        pause_sentence: app.get_pause_sentence(),
        pause_paragraph: app.get_pause_paragraph(),
        auto_save_directory: app.get_auto_save_directory().to_string(),
    }
}

fn load_runtime_config(weak: &slint::Weak<AppWindow>) -> AppConfig {
    weak.upgrade()
        .map(|app| runtime_config(&app))
        .unwrap_or_default()
}

fn set_status(weak: slint::Weak<AppWindow>, message: String) {
    if let Err(error) = slint::invoke_from_event_loop(move || {
        if let Some(app) = weak.upgrade() {
            app.set_status_text(message.into());
        }
    }) {
        eprintln!("Could not update the UI: {error}");
    }
}

fn set_last_generated_file(weak: slint::Weak<AppWindow>, path: String) {
    if let Err(error) = slint::invoke_from_event_loop(move || {
        if let Some(app) = weak.upgrade() {
            app.set_last_generated_file(path.into());
        }
    }) {
        eprintln!("Could not update the UI: {error}");
    }
}

fn load_model_cards(
    weak: slint::Weak<AppWindow>,
    query: String,
    category: String,
    max_size: String,
    language: String,
    framework: String,
    hardware: String,
) {
    let generation = MODEL_SEARCH_GENERATION.fetch_add(1, Ordering::Relaxed) + 1;
    set_status(weak.clone(), "Loading model catalog…".into());
    thread::spawn(move || {
        thread::sleep(Duration::from_millis(250));
        if generation != MODEL_SEARCH_GENERATION.load(Ordering::Relaxed) {
            return;
        }
        let models = if category == "installed" {
            tts::models::installed_models()
        } else {
            tts::models::discover_models(&query, &category)
        };
        if generation != MODEL_SEARCH_GENERATION.load(Ordering::Relaxed) {
            return;
        }
        match models {
            Ok(models) => {
                match filter_models(
                    models, &query, &category, &max_size, &language, &framework, &hardware,
                ) {
                    Ok(models) => set_model_cards(weak, models, generation),
                    Err(message) => set_model_search_status(weak, generation, message),
                }
            }
            Err(error) => {
                let message = format!("Could not load the model catalog: {error}");
                match tts::models::installed_models() {
                    Ok(installed) => {
                        if generation != MODEL_SEARCH_GENERATION.load(Ordering::Relaxed) {
                            return;
                        }
                        let installed = installed
                            .into_iter()
                            .filter(|model| {
                                category == "all"
                                    || category == "languages"
                                    || (category == "tts" && model.task == "text-to-speech")
                                    || (category == "stt"
                                        && model.task == "automatic-speech-recognition")
                            })
                            .collect();
                        match filter_models(
                            installed, &query, &category, &max_size, &language, &framework,
                            &hardware,
                        ) {
                            Ok(installed) => set_model_cards(weak.clone(), installed, generation),
                            Err(filter_error) => {
                                set_model_search_status(weak.clone(), generation, filter_error)
                            }
                        }
                    }
                    Err(scan_error) => eprintln!("Could not scan installed models: {scan_error}"),
                }
                set_model_search_status(weak, generation, message);
            }
        }
    });
}

fn filter_models(
    models: Vec<tts::models::HubModel>,
    query: &str,
    category: &str,
    max_size: &str,
    language: &str,
    framework: &str,
    hardware: &str,
) -> std::result::Result<Vec<tts::models::HubModel>, String> {
    let max_bytes = if max_size.trim().is_empty() {
        None
    } else {
        Some(
            max_size
                .trim()
                .parse::<u64>()
                .map_err(|_| "Maximum model size must be a number of megabytes.".to_owned())?
                .saturating_mul(1024 * 1024),
        )
    };
    let query = query.trim().to_lowercase();
    let language = language.to_lowercase();
    let framework = framework.to_lowercase();
    let hardware = hardware.to_lowercase();
    Ok(models
        .into_iter()
        .filter(|model| {
            let query_matches = model_matches_query(model, &query);
            let language_matches =
                language.is_empty() || model.languages.to_lowercase().contains(&language);
            let framework_matches =
                framework.is_empty() || model.framework.to_lowercase().contains(&framework);
            let hardware_matches =
                hardware.is_empty() || model.compatibility.to_lowercase().contains(&hardware);
            let size_matches =
                max_bytes.is_none_or(|maximum| model.size_bytes > 0 && model.size_bytes <= maximum);
            let category_matches =
                category != "languages" || has_language_metadata(&model.languages);
            query_matches
                && category_matches
                && language_matches
                && framework_matches
                && hardware_matches
                && size_matches
        })
        .collect())
}

fn model_matches_query(model: &tts::models::HubModel, query: &str) -> bool {
    let query = query.to_lowercase();
    let terms = query.split_whitespace().collect::<Vec<_>>();
    terms.is_empty()
        || terms.iter().all(|term| {
            [
                &model.id,
                &model.author,
                &model.task,
                &model.description,
                &model.languages,
                &model.framework,
                &model.architecture,
                &model.license,
                &model.license,
            ]
            .iter()
            .any(|field| field.to_lowercase().contains(term))
                || model
                    .tags
                    .iter()
                    .any(|tag| tag.to_lowercase().contains(term))
        })
}

fn has_language_metadata(languages: &str) -> bool {
    !languages.trim().is_empty()
        && !languages.eq_ignore_ascii_case("unknown")
        && !languages.eq_ignore_ascii_case("not specified")
}

fn set_model_cards(
    weak: slint::Weak<AppWindow>,
    models: Vec<tts::models::HubModel>,
    generation: u64,
) {
    let count = models.len();
    let cards = models
        .into_iter()
        .map(|model| HubModel {
            id: model.id.into(),
            author: model.author.into(),
            task: model.task.clone().into(),
            category_label: if model.task == "text-to-speech" {
                "Text-to-Speech (TTS)".into()
            } else {
                "Speech-to-Text (STT)".into()
            },
            description: model.description.into(),
            languages: model.languages.into(),
            framework: model.framework.into(),
            architecture: model.architecture.into(),
            tags: model.tags.into_iter().collect::<Vec<_>>().join(", ").into(),
            size_label: if model.size_bytes == 0 {
                "Unknown".into()
            } else {
                tts::models::format_bytes(model.size_bytes).into()
            },
            stats: format!("Downloads: {} · Likes: {}", model.downloads, model.likes).into(),
            license: model.license.into(),
            compatibility: model.compatibility.into(),
            installed: model.installed,
            deletable: model.deletable,
            local_path: model.local_path.into(),
        })
        .collect::<Vec<_>>();
    if let Err(error) = slint::invoke_from_event_loop(move || {
        if generation != MODEL_SEARCH_GENERATION.load(Ordering::Relaxed) {
            return;
        }
        if let Some(app) = weak.upgrade() {
            app.set_model_cards(slint::ModelRc::new(slint::VecModel::from(cards)));
            app.set_status_text(if count == 0 {
                "No models match these filters.".into()
            } else {
                format!("Loaded {count} model(s).").into()
            });
        }
    }) {
        eprintln!("Could not update model list: {error}");
    }
}

fn set_model_search_status(weak: slint::Weak<AppWindow>, generation: u64, message: String) {
    if let Err(error) = slint::invoke_from_event_loop(move || {
        if generation == MODEL_SEARCH_GENERATION.load(Ordering::Relaxed) {
            if let Some(app) = weak.upgrade() {
                app.set_status_text(message.into());
            }
        }
    }) {
        eprintln!("Could not update model search status: {error}");
    }
}

#[cfg(test)]
mod model_search_tests {
    use super::*;

    fn model() -> tts::models::HubModel {
        tts::models::HubModel {
            id: "voice/model".into(),
            author: "voice".into(),
            task: "text-to-speech".into(),
            description: "A multilingual expressive voice".into(),
            languages: "English, French".into(),
            tags: vec!["language:en".into(), "expressive".into()],
            framework: "pytorch".into(),
            architecture: "VitsModel".into(),
            size_bytes: 0,
            downloads: 0,
            likes: 0,
            license: "apache-2.0".into(),
            compatibility: "Unknown".into(),
            installed: false,
            deletable: false,
            local_path: String::new(),
        }
    }

    #[test]
    fn model_search_matches_description_and_tags_case_insensitively() {
        let model = model();

        assert!(model_matches_query(&model, "EXPRESSIVE english"));
        assert!(model_matches_query(&model, "apache-2.0"));
        assert!(!model_matches_query(&model, "expressive german"));
        assert!(model_matches_query(&model, "  "));
    }
}

fn set_model_download_progress_text(weak: slint::Weak<AppWindow>, message: String) {
    if let Err(error) = slint::invoke_from_event_loop(move || {
        if let Some(app) = weak.upgrade() {
            app.set_model_download_progress_text(message.into());
        }
    }) {
        eprintln!("Could not update model download progress: {error}");
    }
}

fn set_model_download_progress(weak: slint::Weak<AppWindow>, progress: f32) {
    if let Err(error) = slint::invoke_from_event_loop(move || {
        if let Some(app) = weak.upgrade() {
            app.set_model_download_progress(progress);
        }
    }) {
        eprintln!("Could not update model download progress: {error}");
    }
}

fn set_model_download_progress_indeterminate(weak: slint::Weak<AppWindow>, indeterminate: bool) {
    if let Err(error) = slint::invoke_from_event_loop(move || {
        if let Some(app) = weak.upgrade() {
            app.set_model_download_progress_indeterminate(indeterminate);
        }
    }) {
        eprintln!("Could not update model download progress mode: {error}");
    }
}

fn set_active_model_download(weak: slint::Weak<AppWindow>, id: String) {
    if let Err(error) = slint::invoke_from_event_loop(move || {
        if let Some(app) = weak.upgrade() {
            app.set_active_model_download_id(id.into());
        }
    }) {
        eprintln!("Could not update active model download: {error}");
    }
}

fn format_duration(seconds: u64) -> String {
    if seconds >= 3600 {
        format!("{}h {}m", seconds / 3600, seconds % 3600 / 60)
    } else if seconds >= 60 {
        format!("{}m {}s", seconds / 60, seconds % 60)
    } else {
        format!("{seconds}s")
    }
}

fn open_url(url: &str) -> io::Result<()> {
    #[cfg(target_os = "windows")]
    let mut command = Command::new("rundll32");
    #[cfg(target_os = "windows")]
    command.args(["url.dll,FileProtocolHandler", url]);
    #[cfg(target_os = "macos")]
    let mut command = Command::new("open");
    #[cfg(target_os = "macos")]
    command.arg(url);
    #[cfg(all(unix, not(target_os = "macos")))]
    let mut command = Command::new("xdg-open");
    #[cfg(all(unix, not(target_os = "macos")))]
    command.arg(url);
    command.spawn().map(|_| ())
}

fn transcribe_recording(weak: slint::Weak<AppWindow>, path: PathBuf) {
    let Some(app) = weak.upgrade() else {
        audio::remove_recording(&path);
        return;
    };
    let settings = runtime_config(&app);
    drop(app);

    if settings.whisper_model.trim().is_empty() {
        audio::remove_recording(&path);
        set_status(
            weak,
            "Select a whisper.cpp model file to transcribe the recording.".into(),
        );
        return;
    }
    let model_path = PathBuf::from(settings.whisper_model.clone());
    if !model_path.is_file() {
        audio::remove_recording(&path);
        set_status(
            weak,
            format!(
                "Whisper model file does not exist: {}",
                model_path.display()
            ),
        );
        return;
    }

    set_status(weak.clone(), "Transcribing recording…".into());
    thread::spawn(move || {
        let result = WhisperCliRecognizer.transcribe(&model_path, &path, &settings.language);
        audio::remove_recording(&path);
        match result {
            Ok(text) if text.trim().is_empty() => {
                set_status(weak, "No speech was recognized.".into());
            }
            Ok(text) => {
                if let Err(error) = slint::invoke_from_event_loop(move || {
                    if let Some(app) = weak.upgrade() {
                        app.set_transcript_text(text.into());
                        app.set_status_text("Transcription finished.".into());
                    }
                }) {
                    eprintln!("Could not update the UI: {error}");
                }
            }
            Err(error) => set_status(weak, error.to_string()),
        }
    });
}
