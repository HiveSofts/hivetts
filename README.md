# hivetts
Text to Speech with hivetts


HiveTTS is a local desktop text-to-speech and speech-to-text app built with Slint.
Audio and text are processed locally; no cloud service or account is required.

## Requirements

- Rust and Cargo
- FFmpeg, available on `PATH`, for microphone recording
- On Linux, eSpeak NG (`espeak-ng`) for text-to-speech
- whisper.cpp (`whisper-cli` on `PATH`) and a local whisper.cpp model (`.bin`)
  for speech-to-text

On macOS, the built-in `say` command is used for speech synthesis. FFmpeg uses
the default microphone (`:0`) through its AVFoundation input.

On Linux, FFmpeg records from the default PulseAudio-compatible input. If your
system uses a different input, configure FFmpeg/PulseAudio so `default` refers
to the desired microphone.

Set `WHISPER_MODEL` to a model file path to pre-fill the model field when
launching the app, or enter the path in the UI. For example, after installing
whisper.cpp and downloading a model:

```sh
cargo run
```

Press **Start recording**, speak, then press **Stop recording** to transcribe.
Use **Speak** to read the text currently in the editor aloud. Missing local
tools and model files are reported in the app status area.
