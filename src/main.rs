slint::include_modules!();

use serde::Deserialize;
use std::{
    env,
    io::{self, Write},
    path::PathBuf,
    process::Command,
    thread,
};
use tts::{
    audio,
    config::{self, AppConfig},
    stt::{SpeechRecognizer, WhisperCliRecognizer},
    tts::{LocalSynthesizer, SpeechSynthesizer},
};

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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let app = AppWindow::new()?;
    // Language names and ISO codes are from iso-codes (LGPL-2.1-or-later).
    let language_catalog =
        serde_json::from_str::<Vec<LanguageOptionData>>(include_str!("../ui/languages.json"))?;
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
    app.on_search_languages(move |query| {
        let query = query.trim().to_lowercase();
        let options = language_catalog
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
        thread::spawn(move || {
            let settings = load_runtime_config(&weak);
            let result = LocalSynthesizer.speak(&text, &settings);
            set_status(
                weak,
                result
                    .map(|()| "Speech finished.".to_string())
                    .unwrap_or_else(|error| error.to_string()),
            );
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
    app.on_download_hub_model(move |id, task| {
        let weak = weak.clone();
        let id = id.to_string();
        let task = task.to_string();
        set_active_model_download(weak.clone(), id.clone());
        set_model_download_progress_text(weak.clone(), "Preparing download…".into());
        set_model_download_progress(weak.clone(), 0.0);
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
                    let percent = if total == 0 {
                        "Progress unavailable".to_owned()
                    } else {
                        format!("{}%", (percentage * 100.0).round() as u32)
                    };
                    let text = format!(
                        "{} · {} / {} · {}/s{} · {}",
                        percent,
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
                }
                Err(error) => {
                    set_active_model_download(weak.clone(), String::new());
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
    app.on_delete_model(move |model| {
        let weak = weak.clone();
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
            Ok(()) => set_status(weak.clone(), "Settings saved locally.".into()),
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
        theme: app.get_theme().to_string(),
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

fn load_model_cards(
    weak: slint::Weak<AppWindow>,
    query: String,
    category: String,
    max_size: String,
    language: String,
    framework: String,
    hardware: String,
) {
    set_status(weak.clone(), "Loading model catalog…".into());
    thread::spawn(move || {
        let models = if category == "installed" {
            tts::models::installed_models()
        } else {
            tts::models::discover_models(&query, &category)
        };
        match models {
            Ok(models) => {
                match filter_models(
                    models, &query, &category, &max_size, &language, &framework, &hardware,
                ) {
                    Ok(models) => set_model_cards(weak, models),
                    Err(message) => set_status(weak, message),
                }
            }
            Err(error) => {
                let message = format!("Could not load the model catalog: {error}");
                match tts::models::installed_models() {
                    Ok(installed) => {
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
                            Ok(installed) => set_model_cards(weak.clone(), installed),
                            Err(filter_error) => set_status(weak.clone(), filter_error),
                        }
                    }
                    Err(scan_error) => eprintln!("Could not scan installed models: {scan_error}"),
                }
                set_status(weak, message);
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
    let query = query.to_lowercase();
    let language = language.to_lowercase();
    let framework = framework.to_lowercase();
    let hardware = hardware.to_lowercase();
    Ok(models
        .into_iter()
        .filter(|model| {
            let query_matches = query.is_empty()
                || [
                    &model.id,
                    &model.author,
                    &model.task,
                    &model.languages,
                    &model.architecture,
                    &model.framework,
                ]
                .iter()
                .any(|field| field.to_lowercase().contains(&query));
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

fn has_language_metadata(languages: &str) -> bool {
    !languages.trim().is_empty()
        && !languages.eq_ignore_ascii_case("unknown")
        && !languages.eq_ignore_ascii_case("not specified")
}

fn set_model_cards(weak: slint::Weak<AppWindow>, models: Vec<tts::models::HubModel>) {
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
            size_label: if model.size_bytes == 0 {
                "Unknown".into()
            } else {
                tts::models::format_bytes(model.size_bytes).into()
            },
            stats: if model.downloads == 0 && model.likes == 0 {
                "Usage stats unavailable".into()
            } else {
                format!("{} downloads · {} likes", model.downloads, model.likes).into()
            },
            license: model.license.into(),
            compatibility: model.compatibility.into(),
            installed: model.installed,
            deletable: model.deletable,
            local_path: model.local_path.into(),
        })
        .collect::<Vec<_>>();
    if let Err(error) = slint::invoke_from_event_loop(move || {
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
