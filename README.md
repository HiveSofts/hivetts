# HiveTTS

HiveTTS is a local-first desktop application for text-to-speech and speech
transcription. It uses Slint for the UI and Rust modules for settings, speech
engine adapters, model downloads, and subtitle formatting. Speech inputs and
settings are not sent to a HiveTTS cloud service.

## Architecture and project tree

```text
.
├── Cargo.toml
├── build.rs
├── README.md
├── src/
│   ├── main.rs       # Slint callbacks; long-running work runs on worker threads
│   ├── lib.rs        # Shared library module exports
│   ├── audio.rs      # Microphone recording and temporary audio lifecycle
│   ├── config.rs     # JSON settings and text-file input
│   ├── error.rs      # Shared application error type
│   ├── formats.rs    # SRT/VTT timestamp formatting
│   ├── models.rs     # Local Whisper model list/download/delete helpers
│   ├── stt.rs        # SpeechRecognizer trait and whisper.cpp CLI adapter
│   └── tts.rs        # SpeechSynthesizer trait, Piper CLI and native adapters
└── ui/
    └── app.slint     # Navigation, TTS/STT forms, model actions, and settings
```

The UI calls into `SpeechSynthesizer` and `SpeechRecognizer` implementations.
Inference, model downloads, file reads, and speech playback are started on
background threads; results are posted back to Slint through
`slint::invoke_from_event_loop`.

## Current feature set

- Text editor with Unicode character and word counts, plus loading UTF-8 text
  files by path.
- Piper CLI speech and WAV export when a Piper executable and `.onnx` voice are
  configured; otherwise, OS speech is used where implemented (macOS `say`,
  Linux eSpeak NG/eSpeak).
- Speech-rate control and a runtime light/dark theme toggle.
- Microphone recording on Linux (PulseAudio input) and macOS (AVFoundation),
  recorded as temporary 16 kHz mono WAV and transcribed when recording stops.
- Transcription of audio paths supported by the installed `whisper-cli` build.
- Whisper.cpp tiny, base, and small model downloads; settings are stored as
  local JSON.
- Hugging Face discovery for TTS and automatic speech-recognition repositories,
  with repository metadata, file-size estimates, language/framework filters,
  hardware caveats, and a locally scanned installed-model view.
- Asynchronous Hugging Face downloads with byte-based progress, speed/ETA,
  retry/resume from retained partial files, repository revision tracking, and
  task-organized storage under the local HiveTTS models directory.
- SRT and VTT timestamp formatting utilities with unit tests.

## Current limitations

This repository is an initial, working foundation rather than a complete
implementation of every requested feature. In particular:

- Speech engines currently integrate with local command-line programs; the
  project does not yet use the `whisper-rs`, `cpal`, `rodio`, or Rust `tts`
  bindings. Microphone capture depends on FFmpeg, and Windows microphone
  capture is not implemented.
- Hugging Face downloads can be cancelled and resumed by selecting Download
  again, but do not currently provide a separate pause control. Compatibility is
  conservative: it reports known RAM limitations and otherwise marks hardware
  compatibility unknown rather than claiming support.
- Playback pause/resume/stop, a live input level meter, clipboard actions,
  MP3/SRT/VTT export from the UI, timestamped transcription output, and
  cancellation are not yet wired into the interface.
- Voice selection is configured by a Piper model path. Voice catalogs,
  gender/quality metadata, pitch and volume controls are not yet provided.
- The UI currently contains English strings only; Persian translation and
  complete RTL layout are not yet implemented. The theme setting is persisted,
  and the UI supports a runtime light/dark toggle; automatic system-theme
  detection is not implemented.

These are explicit implementation boundaries, not cloud fallbacks. To add a
Rust-native engine, implement the corresponding trait in `src/tts.rs` or
`src/stt.rs`, register it in `src/main.rs`, and keep model work off the Slint
event loop.

## Requirements and system dependencies

- Rust stable and Cargo
- FFmpeg on `PATH` for microphone capture; the Linux build needs PulseAudio
  input support and the macOS build uses AVFoundation.
- `whisper-cli` from whisper.cpp on `PATH` for speech recognition.
- A local whisper.cpp model (`.bin`) or use the Models tab to download tiny,
  base, or small. The model manager requires `curl`; set `HF_TOKEN` in the
  process environment when accessing gated Hugging Face repositories.
- Linux TTS fallback: install eSpeak NG (`espeak-ng`) or eSpeak. Piper is
  optional.
- Piper speech and WAV export: install the Piper executable and download a
  compatible voice model (`.onnx`); set both paths in Settings. The voice's
  companion config file must be available as expected by the Piper release.
- Linux UI prerequisites vary by Slint backend and distribution. If Cargo
  reports missing system libraries, install the distribution packages for
  X11/Wayland, fontconfig, and audio development headers.

### Installing external engines

Build whisper.cpp using its upstream build instructions and ensure its
`whisper-cli` executable is on `PATH`. Download a compatible GGML model, or
select a model in HiveTTS's Models tab. For Piper, install its upstream
executable and a voice `.onnx` file, then enter their paths in the Settings
tab. The app does not automatically fetch Piper voice files.

For manual model selection, set `WHISPER_MODEL` before launching, or enter the
model path in Settings. The Models tab downloads whisper.cpp's `ggml-tiny.bin`,
`ggml-base.bin`, or `ggml-small.bin` into the app's local data directory.

## Build and run

All platforms:

```sh
cargo fmt --all -- --check
cargo test
cargo run
```

### Linux

Install Rust, FFmpeg (built with PulseAudio support), and eSpeak NG if you want
the native fallback. Install the Slint UI system development libraries required
by your distribution (typically X11 or Wayland, fontconfig, and audio headers).
For microphone capture, configure the PulseAudio default source.

### macOS

Install Rust and FFmpeg. The built-in `say` command is used for native speech;
FFmpeg uses AVFoundation's default input (`:0`) for microphone capture.

### Windows

Install Rust and FFmpeg to build the application. Speech playback uses the
Windows PowerShell speech synthesizer fallback; Piper can be configured for
speech and WAV export. Microphone capture is currently unsupported on Windows.

## Usage

1. Launch HiveTTS and select a tab.
2. In **Text to Speech**, type or load text, set an output WAV path for export,
   then choose Speak or Export WAV.
3. In **Speech to Text**, enter a model in Settings, then start/stop microphone
   recording or enter an audio file path and select Transcribe file.
4. Use **Models** to discover TTS/ASR repositories from Hugging Face, filter
   results, download models, and manage installed model files. The Whisper.cpp
   tiny/base/small downloads remain available in the same section.
5. Adjust engine paths, language (`auto` or a language code), and speech rate in
   **Settings**. Select Save settings to persist configuration.

Settings are saved in `settings.json` under the user's HiveTTS configuration
directory. Temporary microphone recordings are removed after transcription.
