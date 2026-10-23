slint::include_modules!();

use std::{
    env,
    io::{self, Write},
    path::PathBuf,
    thread,
};
use tts::{
    audio,
    config::{self, AppConfig},
    stt::{SpeechRecognizer, WhisperCliRecognizer},
    tts::{LocalSynthesizer, SpeechSynthesizer},
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let app = AppWindow::new()?;
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
