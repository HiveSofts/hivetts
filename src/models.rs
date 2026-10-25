use crate::error::{AppError, Result};
use serde::{Deserialize, Serialize};
use std::{
    env, fs,
    io::Write,
    path::{Component, Path, PathBuf},
    process::{Command, Stdio},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex, OnceLock,
    },
    thread,
    time::{Duration, Instant},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WhisperSize {
    Tiny,
    Base,
    Small,
}

impl WhisperSize {
    pub fn filename(self) -> &'static str {
        match self {
            Self::Tiny => "ggml-tiny.bin",
            Self::Base => "ggml-base.bin",
            Self::Small => "ggml-small.bin",
        }
    }

    fn url(self) -> String {
        format!(
            "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/{}",
            self.filename()
        )
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HubModel {
    pub id: String,
    pub author: String,
    pub task: String,
    pub description: String,
    pub languages: String,
    pub tags: Vec<String>,
    pub framework: String,
    pub architecture: String,
    pub size_bytes: u64,
    pub downloads: u64,
    pub likes: u64,
    pub license: String,
    pub compatibility: String,
    pub installed: bool,
    pub deletable: bool,
    pub local_path: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
struct RegistryEntry {
    id: String,
    task: String,
    revision: String,
    local_path: String,
    files: Vec<String>,
    file_sizes: std::collections::HashMap<String, u64>,
    size_bytes: u64,
    author: String,
    description: String,
    languages: String,
    tags: Vec<String>,
    framework: String,
    architecture: String,
    license: String,
    downloads: u64,
    likes: u64,
}

impl Default for RegistryEntry {
    fn default() -> Self {
        Self {
            id: String::new(),
            task: String::new(),
            revision: String::new(),
            local_path: String::new(),
            files: Vec::new(),
            file_sizes: std::collections::HashMap::new(),
            size_bytes: 0,
            author: String::new(),
            description: String::new(),
            languages: String::new(),
            tags: Vec::new(),
            framework: String::new(),
            architecture: String::new(),
            license: String::new(),
            downloads: 0,
            likes: 0,
        }
    }
}
#[derive(Deserialize)]
struct ApiModel {
    id: String,
    #[serde(default)]
    author: Option<String>,
    #[serde(default)]
    downloads: u64,
    #[serde(default)]
    likes: u64,
    #[serde(default)]
    sha: Option<String>,
    #[serde(default)]
    pipeline_tag: Option<String>,
    #[serde(default)]
    tags: Vec<String>,
    #[serde(default)]
    library_name: Option<String>,
    #[serde(default)]
    siblings: Vec<ApiSibling>,
    #[serde(default)]
    config: Option<serde_json::Value>,
    #[serde(default, rename = "cardData")]
    card_data: Option<serde_json::Value>,
}

#[derive(Deserialize)]
struct ApiSibling {
    rfilename: String,
    #[serde(default)]
    size: Option<u64>,
    #[serde(default)]
    lfs: Option<ApiLfs>,
}

#[derive(Deserialize)]
struct ApiLfs {
    size: u64,
}

pub fn models_directory() -> Result<PathBuf> {
    let base = if cfg!(target_os = "windows") {
        env::var_os("LOCALAPPDATA").map(PathBuf::from)
    } else if cfg!(target_os = "macos") {
        env::var_os("HOME").map(|home| PathBuf::from(home).join("Library/Application Support"))
    } else {
        env::var_os("XDG_DATA_HOME")
            .map(PathBuf::from)
            .or_else(|| env::var_os("HOME").map(|home| PathBuf::from(home).join(".local/share")))
    }
    .ok_or_else(|| AppError::Message("Could not locate the user data directory.".into()))?;
    Ok(base.join("hivetts").join("models"))
}

pub fn list_models() -> Result<Vec<PathBuf>> {
    let directory = models_directory()?;
    if !directory.exists() {
        return Ok(Vec::new());
    }
    let mut models = fs::read_dir(directory)?
        .filter_map(std::result::Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "bin"))
        .collect::<Vec<_>>();
    models.sort();
    Ok(models)
}

pub fn download_model(size: WhisperSize) -> Result<PathBuf> {
    let directory = models_directory()?;
    fs::create_dir_all(&directory)?;
    let destination = directory.join(size.filename());
    if destination.is_file() {
        return Ok(destination);
    }
    let temporary = directory.join(format!("{}.part-{}", size.filename(), std::process::id()));
    let curl = find_executable("curl")
        .ok_or_else(|| AppError::Message("Install curl to download speech models.".into()))?;
    let status = Command::new(curl)
        .args(["--fail", "--location", "--progress-bar", "--output"])
        .arg(&temporary)
        .arg(size.url())
        .status()?;
    if !status.success() {
        if let Err(error) = fs::remove_file(&temporary) {
            if error.kind() != std::io::ErrorKind::NotFound {
                eprintln!("Could not remove partial model download: {error}");
            }
        }
        return Err(AppError::Message(format!(
            "Model download failed with status {status}."
        )));
    }
    fs::rename(&temporary, &destination)?;
    Ok(destination)
}

pub fn discover_models(query: &str, category: &str) -> Result<Vec<HubModel>> {
    let mut models = Vec::new();
    let tasks = match category {
        "tts" => vec!["text-to-speech"],
        "stt" => vec!["automatic-speech-recognition"],
        "languages" => vec!["text-to-speech", "automatic-speech-recognition"],
        _ => vec!["text-to-speech", "automatic-speech-recognition"],
    };
    let installed = installed_registry()?;
    for task in tasks {
        let mut url = format!(
            "https://huggingface.co/api/models?pipeline_tag={task}&sort=downloads&direction=-1&limit=30&full=true"
        );
        if !query.trim().is_empty() {
            url.push_str("&search=");
            url.push_str(&encode_query(query.trim()));
        }
        let response = curl_output(&[&url])?;
        let mut listed: Vec<ApiModel> = serde_json::from_slice(&response)?;
        if !query.trim().is_empty() {
            let tagged_candidates_url = format!(
                "https://huggingface.co/api/models?pipeline_tag={task}&sort=downloads&direction=-1&limit=12&full=true"
            );
            let tagged_candidates: Vec<ApiModel> =
                serde_json::from_slice(&curl_output(&[&tagged_candidates_url])?)?;
            let mut seen = listed
                .iter()
                .map(|model| model.id.clone())
                .collect::<std::collections::HashSet<_>>();
            listed.extend(tagged_candidates.into_iter().filter(|model| {
                api_model_matches_query(model, query) && seen.insert(model.id.clone())
            }));
        }
        for listed_model in listed {
            let details_url = format!(
                "https://huggingface.co/api/models/{}?blobs=true",
                encode_repo(&listed_model.id)
            );
            let details = match curl_output(&[&details_url]) {
                Ok(response) => serde_json::from_slice::<ApiModel>(&response).ok(),
                Err(error) => {
                    eprintln!(
                        "Detailed metadata unavailable for {}: {error}",
                        listed_model.id
                    );
                    None
                }
            };
            let metadata = details.as_ref().unwrap_or(&listed_model);
            let file_size = metadata
                .siblings
                .iter()
                .map(|file| {
                    file.size
                        .or_else(|| file.lfs.as_ref().map(|lfs| lfs.size))
                        .unwrap_or(0)
                })
                .sum();
            let id = listed_model.id.clone();
            let record = installed.get(&id);
            let is_installed = record.is_some_and(|entry| verify_entry(entry));
            let model_task = metadata
                .pipeline_tag
                .as_deref()
                .or(listed_model.pipeline_tag.as_deref())
                .unwrap_or(task);
            let author = metadata
                .author
                .as_deref()
                .or(listed_model.author.as_deref())
                .map(str::to_owned)
                .unwrap_or_else(|| id.split('/').next().unwrap_or("unknown").to_owned());
            let languages = model_languages(metadata);
            let architecture = model_architecture(metadata);
            let mut tags = model_tags(metadata);
            for tag in model_tags(&listed_model) {
                if !tags.contains(&tag) {
                    tags.push(tag);
                }
            }
            let description = model_description(metadata).unwrap_or_else(|| {
                readme_description(&id)
                    .unwrap_or_else(|| "Description not provided by the model repository.".into())
            });
            let compatibility = assess_compatibility(file_size);
            models.push(HubModel {
                id: id.clone(),
                author,
                task: model_task.to_owned(),
                description,
                languages: if languages.is_empty() {
                    "Not specified".into()
                } else {
                    languages.join(", ")
                },
                tags,
                framework: metadata
                    .library_name
                    .clone()
                    .unwrap_or_else(|| "Not specified".into()),
                architecture,
                size_bytes: if is_installed {
                    record.map(|entry| entry.size_bytes).unwrap_or(file_size)
                } else {
                    file_size
                },
                downloads: metadata.downloads,
                likes: metadata.likes,
                license: model_license(metadata),
                compatibility,
                installed: is_installed,
                deletable: is_installed,
                local_path: record
                    .map(|entry| entry.local_path.clone())
                    .unwrap_or_default(),
            });
        }
    }
    Ok(models)
}

pub fn installed_models() -> Result<Vec<HubModel>> {
    let entries = installed_registry()?;
    let mut models = entries
        .values()
        .filter(|entry| verify_entry(entry))
        .map(|entry| HubModel {
            id: entry.id.clone(),
            author: display_or_unknown(&entry.author),
            task: entry.task.clone(),
            description: display_or_unknown(&entry.description),
            languages: display_or_unknown(&entry.languages),
            tags: entry.tags.clone(),
            framework: display_or_unknown(&entry.framework),
            architecture: display_or_unknown(&entry.architecture),
            size_bytes: entry.size_bytes,
            downloads: entry.downloads,
            likes: entry.likes,
            license: display_or_unknown(&entry.license),
            compatibility: assess_compatibility(entry.size_bytes),
            installed: true,
            deletable: true,
            local_path: entry.local_path.clone(),
        })
        .collect::<Vec<_>>();
    for path in list_models()? {
        let Some(filename) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        models.push(HubModel {
            id: format!("Whisper.cpp · {filename}"),
            author: "ggerganov".into(),
            task: "automatic-speech-recognition".into(),
            description: "Legacy whisper.cpp model already present in local storage.".into(),
            languages: "Multilingual".into(),
            tags: vec!["automatic-speech-recognition".into(), "whisper".into()],
            framework: "whisper.cpp".into(),
            architecture: "Whisper GGML".into(),
            size_bytes: fs::metadata(&path)?.len(),
            downloads: 0,
            likes: 0,
            license: "Check upstream repository".into(),
            compatibility: assess_compatibility(fs::metadata(&path)?.len()),
            installed: true,
            deletable: false,
            local_path: path.to_string_lossy().into_owned(),
        });
    }
    models.sort_by(|left, right| {
        left.task
            .cmp(&right.task)
            .then_with(|| left.id.cmp(&right.id))
    });
    Ok(models)
}

pub fn download_hub_model(
    id: &str,
    task: &str,
    mut progress: impl FnMut(u64, u64, &str, f64),
) -> Result<PathBuf> {
    let active = active_hub_download();
    let cancel = Arc::new(AtomicBool::new(false));
    {
        let mut active = active.lock().map_err(|error| {
            AppError::Message(format!("Download state is unavailable: {error}"))
        })?;
        if active.is_some() {
            return Err(AppError::Message(
                "Another model download is already active.".into(),
            ));
        }
        *active = Some((id.to_owned(), cancel.clone()));
    }
    let result = download_hub_model_inner(id, task, &mut progress, &cancel);
    if let Ok(mut active) = active.lock() {
        *active = None;
    }
    result
}

pub fn cancel_hub_download(id: &str) -> Result<()> {
    let active = active_hub_download()
        .lock()
        .map_err(|error| AppError::Message(format!("Download state is unavailable: {error}")))?;
    match active.as_ref() {
        Some((active_id, cancel)) if active_id == id => {
            cancel.store(true, Ordering::Release);
            Ok(())
        }
        _ => Err(AppError::Message(
            "That model is not currently downloading.".into(),
        )),
    }
}

fn active_hub_download() -> &'static Mutex<Option<(String, Arc<AtomicBool>)>> {
    static ACTIVE: OnceLock<Mutex<Option<(String, Arc<AtomicBool>)>>> = OnceLock::new();
    ACTIVE.get_or_init(|| Mutex::new(None))
}

fn download_hub_model_inner(
    id: &str,
    task: &str,
    progress: &mut impl FnMut(u64, u64, &str, f64),
    cancel: &AtomicBool,
) -> Result<PathBuf> {
    if !matches!(task, "text-to-speech" | "automatic-speech-recognition") {
        return Err(AppError::Message("Unsupported model task.".into()));
    }
    let details_url = format!(
        "https://huggingface.co/api/models/{}?blobs=true",
        encode_repo(id)
    );
    let model: ApiModel = serde_json::from_slice(&curl_output(&[&details_url])?)?;
    let revision = model.sha.clone().unwrap_or_else(|| "main".into());
    if model.siblings.is_empty() {
        return Err(AppError::Message(
            "The repository did not provide a downloadable file list.".into(),
        ));
    }
    let root = models_directory()?;
    let task_dir = if task == "text-to-speech" {
        "tts"
    } else {
        "stt"
    };
    let destination = root.join(task_dir).join(safe_repo_name(id));
    fs::create_dir_all(&destination)?;
    let canonical_root = root.canonicalize()?;
    let canonical_destination = destination.canonicalize()?;
    if !canonical_destination.starts_with(&canonical_root) {
        return Err(AppError::Message(
            "The model installation folder resolves outside model storage.".into(),
        ));
    }
    let in_progress = destination.join(".download-in-progress");
    let files = model
        .siblings
        .iter()
        .filter(|file| safe_relative_path(&file.rfilename))
        .collect::<Vec<_>>();
    if files.is_empty() {
        return Err(AppError::Message(
            "The repository contains no safe files to download.".into(),
        ));
    }
    let declared_total = files
        .iter()
        .map(|file| {
            file.size
                .or_else(|| file.lfs.as_ref().map(|lfs| lfs.size))
                .unwrap_or(0)
        })
        .sum::<u64>();
    let total = if files.iter().any(|file| {
        file.size
            .or_else(|| file.lfs.as_ref().map(|lfs| lfs.size))
            .unwrap_or(0)
            == 0
    }) {
        0
    } else {
        declared_total
    };
    let remaining_space = files
        .iter()
        .map(|file| {
            let file_size = file
                .size
                .or_else(|| file.lfs.as_ref().map(|lfs| lfs.size))
                .unwrap_or(0);
            if file_size == 0 {
                return 0;
            }
            let target = destination.join(&file.rfilename);
            if target.is_file()
                && fs::metadata(&target).is_ok_and(|metadata| metadata.len() == file_size)
            {
                return 0;
            }
            let temporary = temporary_download_path(&target);
            let partial_size = fs::metadata(temporary)
                .map(|metadata| metadata.len())
                .unwrap_or(0);
            file_size.saturating_sub(partial_size)
        })
        .sum();
    check_disk_space(&root, remaining_space)?;
    let curl = find_executable("curl")
        .ok_or_else(|| AppError::Message("Install curl to download models.".into()))?;
    fs::write(&in_progress, b"")?;
    let token = env::var("HF_TOKEN").ok();
    let mut completed = 0u64;
    let start = Instant::now();
    let mut installed_files = Vec::new();
    let mut file_sizes = std::collections::HashMap::new();
    for file in &files {
        if cancel.load(Ordering::Acquire) {
            return Err(AppError::Message(
                "Download cancelled. Partial files are retained; select Download again to resume."
                    .into(),
            ));
        }
        let file_size = file
            .size
            .or_else(|| file.lfs.as_ref().map(|lfs| lfs.size))
            .unwrap_or(0);
        let relative = Path::new(&file.rfilename);
        let target = destination.join(relative);
        let parent = target
            .parent()
            .ok_or_else(|| AppError::Message("Invalid model file path.".into()))?;
        fs::create_dir_all(parent)?;
        if !parent.canonicalize()?.starts_with(&canonical_destination) || target.is_symlink() {
            return Err(AppError::Message(
                "A model file path resolves outside its installation directory.".into(),
            ));
        }
        if target.is_file() && (file_size == 0 || fs::metadata(&target)?.len() == file_size) {
            let actual_size = fs::metadata(&target)?.len();
            completed += actual_size;
            installed_files.push(file.rfilename.clone());
            file_sizes.insert(file.rfilename.clone(), actual_size);
            continue;
        }
        let temporary = temporary_download_path(&target);
        let encoded_path = file
            .rfilename
            .split('/')
            .map(encode_query)
            .collect::<Vec<_>>()
            .join("/");
        let url = format!(
            "https://huggingface.co/{}/resolve/{}/{}",
            id, revision, encoded_path
        );
        let redirect_url = if let Some(token) = token.as_deref() {
            authenticated_redirect_url(&url, token)?
        } else {
            None
        };
        let transfer_url = redirect_url.as_deref().unwrap_or(&url);
        let use_auth = token.is_some() && redirect_url.is_none();
        let mut command = Command::new(&curl);
        command.args(["--fail"]);
        if !use_auth {
            command.arg("--location");
        }
        command.args([
            "--retry",
            "2",
            "--continue-at",
            "-",
            "--silent",
            "--show-error",
            "--output",
        ]);
        command
            .arg(&temporary)
            .arg(transfer_url)
            .stdout(Stdio::null());
        let mut child = spawn_curl(command, if use_auth { token.as_deref() } else { None })?;
        loop {
            if cancel.load(Ordering::Acquire) {
                child.kill()?;
                child.wait()?;
                return Err(AppError::Message(
                    "Download cancelled. Partial files are retained; select Download again to resume."
                        .into(),
                ));
            }
            let current = fs::metadata(&temporary)
                .map(|metadata| metadata.len())
                .unwrap_or(0);
            let elapsed = start.elapsed().as_secs_f64().max(0.001);
            progress(
                completed.saturating_add(current),
                total,
                &file.rfilename,
                completed.saturating_add(current) as f64 / elapsed,
            );
            if let Some(status) = child.try_wait()? {
                if !status.success() {
                    let detail = if status.code() == Some(23) {
                        "Disk write failed; check free space and file permissions.".to_owned()
                    } else {
                        format!(
                            "HTTP/transfer status {status}. Check repository access and HF_TOKEN for gated models."
                        )
                    };
                    return Err(AppError::Message(format!(
                        "Download failed for {}. {detail} Partial data is retained for retry.",
                        file.rfilename
                    )));
                }
                break;
            }
            thread::sleep(Duration::from_millis(250));
        }
        if file_size > 0 && fs::metadata(&temporary)?.len() != file_size {
            let _ = fs::remove_file(&temporary);
            return Err(AppError::Message(format!(
                "Downloaded size verification failed for {}.",
                file.rfilename
            )));
        }
        fs::rename(&temporary, &target)?;
        completed += file_size.max(fs::metadata(&target)?.len());
        installed_files.push(file.rfilename.clone());
        file_sizes.insert(file.rfilename.clone(), fs::metadata(&target)?.len());
        progress(
            completed,
            total,
            &file.rfilename,
            completed as f64 / start.elapsed().as_secs_f64().max(0.001),
        );
    }
    let entry = RegistryEntry {
        id: id.to_owned(),
        task: task.to_owned(),
        revision,
        local_path: destination.to_string_lossy().into_owned(),
        files: installed_files,
        file_sizes,
        size_bytes: directory_size(&destination)?,
        author: model
            .author
            .clone()
            .unwrap_or_else(|| id.split('/').next().unwrap_or("unknown").to_owned()),
        description: model
            .card_data
            .as_ref()
            .and_then(|card| card.get("description"))
            .and_then(serde_json::Value::as_str)
            .unwrap_or("")
            .to_owned(),
        languages: model_languages(&model).join(", "),
        tags: model_tags(&model),
        framework: model.library_name.clone().unwrap_or_default(),
        architecture: model_architecture(&model),
        license: model_license(&model),
        downloads: model.downloads,
        likes: model.likes,
    };
    update_registry(entry)?;
    fs::remove_file(in_progress)?;
    Ok(destination)
}

pub fn open_model_folder(path: &Path) -> Result<()> {
    let canonical = path.canonicalize()?;
    let root = models_directory()?.canonicalize()?;
    let folder = if canonical.is_file() {
        canonical
            .parent()
            .ok_or_else(|| AppError::Message("The model path has no parent directory.".into()))?
            .to_path_buf()
    } else {
        canonical
    };
    if !folder.starts_with(root) || !folder.is_dir() {
        return Err(AppError::Message(
            "The model folder is not inside HiveTTS storage.".into(),
        ));
    }
    #[cfg(target_os = "windows")]
    let mut command = Command::new("explorer");
    #[cfg(target_os = "macos")]
    let mut command = Command::new("open");
    #[cfg(all(unix, not(target_os = "macos")))]
    let mut command = Command::new("xdg-open");
    command.arg(folder).spawn()?;
    Ok(())
}

pub fn delete_hub_model(id: &str) -> Result<()> {
    let entries = installed_registry()?;
    let entry = entries.get(id).ok_or_else(|| {
        AppError::Message("Installed model was not found in the registry.".into())
    })?;
    let path = PathBuf::from(&entry.local_path);
    let root = models_directory()?.canonicalize()?;
    let canonical = path.canonicalize()?;
    if !canonical.starts_with(&root) || canonical == root {
        return Err(AppError::Message(
            "Refusing to delete a path outside model storage.".into(),
        ));
    }
    fs::remove_dir_all(canonical)?;
    let mut updated = entries;
    updated.remove(id);
    write_registry(&updated)?;
    Ok(())
}

fn installed_registry() -> Result<std::collections::HashMap<String, RegistryEntry>> {
    let path = models_directory()?.join("model_registry.json");
    let mut entries: std::collections::HashMap<String, RegistryEntry> = match fs::read(&path) {
        Ok(bytes) => serde_json::from_slice(&bytes)?,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Default::default(),
        Err(error) => return Err(error.into()),
    };
    let root = models_directory()?;
    let mut registry_changed = false;
    for task_dir in ["tts", "stt"] {
        let directory = root.join(task_dir);
        match fs::symlink_metadata(&directory) {
            Ok(metadata) if metadata.file_type().is_dir() => {}
            Ok(_) => continue,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
            Err(error) => return Err(error.into()),
        }
        let models = fs::read_dir(directory)?;
        for model_entry in models {
            let model_entry = model_entry?;
            if !model_entry.file_type()?.is_dir() {
                continue;
            }
            let model_dir = model_entry.path();
            if model_dir.join(".download-in-progress").exists()
                || entries
                    .values()
                    .any(|entry| Path::new(&entry.local_path) == model_dir)
            {
                continue;
            }
            let files = collect_files(&model_dir)?;
            if files.is_empty() {
                continue;
            }
            let file_sizes = files
                .iter()
                .filter_map(|file| {
                    fs::metadata(model_dir.join(file))
                        .ok()
                        .map(|metadata| (file.clone(), metadata.len()))
                })
                .collect();
            let id = model_dir
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("unknown")
                .replace("--", "/");
            entries.insert(
                id.clone(),
                RegistryEntry {
                    id,
                    task: if task_dir == "tts" {
                        "text-to-speech"
                    } else {
                        "automatic-speech-recognition"
                    }
                    .into(),
                    revision: "unknown".into(),
                    local_path: model_dir.to_string_lossy().into_owned(),
                    files,
                    file_sizes,
                    size_bytes: directory_size(&model_dir)?,
                    ..Default::default()
                },
            );
            registry_changed = true;
        }
    }
    if registry_changed {
        write_registry(&entries)?;
    }
    Ok(entries)
}

fn update_registry(entry: RegistryEntry) -> Result<()> {
    let mut entries = installed_registry()?;
    entries.insert(entry.id.clone(), entry);
    write_registry(&entries)
}

fn write_registry(entries: &std::collections::HashMap<String, RegistryEntry>) -> Result<()> {
    let path = models_directory()?.join("model_registry.json");
    let temporary = path.with_extension("json.tmp");
    fs::write(&temporary, serde_json::to_vec_pretty(entries)?)?;
    fs::rename(temporary, path)?;
    Ok(())
}

fn verify_entry(entry: &RegistryEntry) -> bool {
    let path = Path::new(&entry.local_path);
    let Ok(root) = models_directory() else {
        return false;
    };
    let Ok(canonical_root) = root.canonicalize() else {
        return false;
    };
    let Ok(canonical_path) = path.canonicalize() else {
        return false;
    };
    canonical_path.starts_with(canonical_root)
        && fs::symlink_metadata(path).is_ok_and(|metadata| metadata.file_type().is_dir())
        && !entry.files.is_empty()
        && entry.files.iter().all(|file| {
            if !safe_relative_path(file) {
                return false;
            }
            let file_path = path.join(file);
            let Ok(metadata) = fs::symlink_metadata(&file_path) else {
                return false;
            };
            metadata.file_type().is_file()
                && entry
                    .file_sizes
                    .get(file)
                    .is_none_or(|expected| metadata.len() == *expected)
        })
}

fn collect_files(directory: &Path) -> Result<Vec<String>> {
    let mut files = Vec::new();
    for entry in fs::read_dir(directory)? {
        let entry = entry?;
        let path = entry.path();
        let file_type = entry.file_type()?;
        if file_type.is_dir() {
            files.extend(
                collect_files(&path)?
                    .into_iter()
                    .map(|child| format!("{}/{}", entry.file_name().to_string_lossy(), child)),
            );
        } else if file_type.is_file()
            && !path.file_name().is_some_and(|name| {
                name.to_string_lossy().contains(".part") || name == ".download-in-progress"
            })
        {
            files.push(
                path.strip_prefix(directory)
                    .map_err(|error| AppError::Message(error.to_string()))?
                    .to_string_lossy()
                    .into_owned(),
            );
        }
    }
    Ok(files)
}

fn directory_size(path: &Path) -> Result<u64> {
    Ok(collect_files(path)?
        .iter()
        .filter_map(|file| fs::metadata(path.join(file)).ok())
        .map(|metadata| metadata.len())
        .sum())
}

fn check_disk_space(root: &Path, required: u64) -> Result<()> {
    if let Some(available) = available_disk_space(root) {
        if required > available {
            return Err(AppError::Message(format!(
                "Not enough disk space: {} required, {} available.",
                format_bytes(required),
                format_bytes(available)
            )));
        }
    }
    Ok(())
}

fn available_disk_space(path: &Path) -> Option<u64> {
    #[cfg(unix)]
    {
        let mut probe = path;
        while !probe.exists() {
            probe = probe.parent()?;
        }
        let output = Command::new("df").arg("-Pk").arg(probe).output().ok()?;
        if !output.status.success() {
            return None;
        }
        let text = String::from_utf8_lossy(&output.stdout);
        let available_kb = text
            .lines()
            .nth(1)?
            .split_whitespace()
            .nth(3)?
            .parse::<u64>()
            .ok()?;
        Some(available_kb.saturating_mul(1024))
    }
    #[cfg(not(unix))]
    {
        let _ = path;
        None
    }
}

fn temporary_download_path(target: &Path) -> PathBuf {
    target.with_extension(format!(
        "{}part",
        target
            .extension()
            .and_then(|extension| extension.to_str())
            .map(|extension| format!("{extension}."))
            .unwrap_or_default()
    ))
}

fn assess_compatibility(model_size: u64) -> String {
    let platform = format!("{} / {}", env::consts::OS, env::consts::ARCH);
    if model_size > 0 {
        if let Ok(path) = models_directory() {
            if available_disk_space(&path).is_some_and(|available| model_size > available) {
                return format!(
                    "Incompatible — model needs {} but less disk space is available",
                    format_bytes(model_size)
                );
            }
        }
    }
    #[cfg(target_os = "linux")]
    if let Ok(meminfo) = fs::read_to_string("/proc/meminfo") {
        if let Some(available_kb) = meminfo
            .lines()
            .find(|line| line.starts_with("MemAvailable:"))
            .and_then(|line| line.split_whitespace().nth(1))
            .and_then(|value| value.parse::<u64>().ok())
        {
            if model_size > available_kb.saturating_mul(1024) {
                return format!(
                    "Limited compatibility — {} repository files exceed available RAM; inference requirements are unknown",
                    platform
                );
            }
        }
    }
    format!("Unknown — {platform}; runtime and GPU requirements are not declared")
}

fn model_architecture(model: &ApiModel) -> String {
    model
        .config
        .as_ref()
        .and_then(|config| config.get("architectures"))
        .and_then(serde_json::Value::as_array)
        .and_then(|architectures| architectures.first())
        .and_then(serde_json::Value::as_str)
        .or_else(|| {
            model
                .tags
                .iter()
                .find_map(|tag| tag.strip_prefix("architecture:"))
        })
        .or_else(|| model.tags.iter().find_map(|tag| tag.strip_prefix("model:")))
        .unwrap_or("Not specified")
        .to_owned()
}

fn model_languages(model: &ApiModel) -> Vec<String> {
    let tagged = model
        .tags
        .iter()
        .filter_map(|tag| tag.strip_prefix("language:"))
        .map(str::to_owned)
        .collect::<Vec<_>>();
    if !tagged.is_empty() {
        return tagged;
    }
    model
        .card_data
        .as_ref()
        .and_then(|card| card.get("language"))
        .map(|language| match language {
            serde_json::Value::String(language) => vec![language.clone()],
            serde_json::Value::Array(languages) => languages
                .iter()
                .filter_map(serde_json::Value::as_str)
                .map(str::to_owned)
                .collect(),
            _ => Vec::new(),
        })
        .unwrap_or_default()
}

fn api_model_matches_query(model: &ApiModel, query: &str) -> bool {
    let description = model_description(model).unwrap_or_default();
    let fields = [
        model.id.as_str(),
        model.author.as_deref().unwrap_or_default(),
        model.pipeline_tag.as_deref().unwrap_or_default(),
        model.library_name.as_deref().unwrap_or_default(),
        description.as_str(),
    ];
    let languages = model_languages(model).join(" ");
    let architecture = model_architecture(model);
    let license = model_license(model);
    query.split_whitespace().map(str::to_lowercase).all(|term| {
        fields
            .iter()
            .any(|field| field.to_lowercase().contains(&term))
            || [languages.as_str(), architecture.as_str(), license.as_str()]
                .iter()
                .any(|field| field.to_lowercase().contains(&term))
            || model
                .tags
                .iter()
                .any(|tag| tag.to_lowercase().contains(&term))
    })
}

fn model_tags(model: &ApiModel) -> Vec<String> {
    let mut tags = model.tags.clone();
    if let Some(pipeline_tag) = model.pipeline_tag.as_ref() {
        if !tags.contains(pipeline_tag) {
            tags.push(pipeline_tag.clone());
        }
    }
    if let Some(library) = model.library_name.as_deref() {
        if !tags.iter().any(|tag| tag == library) {
            tags.push(library.to_owned());
        }
    }
    tags
}

fn model_license(model: &ApiModel) -> String {
    model
        .tags
        .iter()
        .find_map(|tag| tag.strip_prefix("license:"))
        .or_else(|| {
            model
                .card_data
                .as_ref()
                .and_then(|card| card.get("license"))
                .and_then(serde_json::Value::as_str)
        })
        .unwrap_or("Not specified")
        .to_owned()
}

fn model_description(model: &ApiModel) -> Option<String> {
    model
        .card_data
        .as_ref()
        .and_then(|card| card.get("description"))
        .and_then(serde_json::Value::as_str)
        .filter(|description| !description.trim().is_empty())
        .map(str::to_owned)
}

fn readme_description(id: &str) -> Option<String> {
    let url = format!("https://huggingface.co/{id}/raw/main/README.md");
    let contents = curl_output(&["--range", "0-65535", &url]).ok()?;
    let contents = String::from_utf8_lossy(&contents);
    let mut lines = contents.lines().peekable();
    if lines.peek().is_some_and(|line| line.trim() == "---") {
        lines.next();
        for line in lines.by_ref() {
            if line.trim() == "---" {
                break;
            }
        }
    }
    let mut paragraph = Vec::new();
    for line in lines {
        let line = line.trim();
        if line.is_empty() {
            if !paragraph.is_empty() {
                break;
            }
            continue;
        }
        if line.starts_with('#')
            || line.starts_with("<!--")
            || line.starts_with("[![")
            || line.starts_with("![")
        {
            continue;
        }
        paragraph.push(line);
    }
    if paragraph.is_empty() {
        return None;
    }
    let text = paragraph.join(" ");
    let trimmed = text.chars().take(320).collect::<String>();
    Some(trimmed)
}

fn display_or_unknown(value: &str) -> String {
    if value.trim().is_empty() {
        "Not specified".into()
    } else {
        value.to_owned()
    }
}

pub fn format_bytes(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KB", "MB", "GB", "TB"];
    let mut size = bytes as f64;
    let mut unit = 0;
    while size >= 1024.0 && unit < UNITS.len() - 1 {
        size /= 1024.0;
        unit += 1;
    }
    format!("{size:.1} {}", UNITS[unit])
}

fn curl_output(args: &[&str]) -> Result<Vec<u8>> {
    let curl = find_executable("curl")
        .ok_or_else(|| AppError::Message("Install curl to connect to Hugging Face.".into()))?;
    let mut command = Command::new(curl);
    command
        .args(["--fail", "--silent", "--show-error"])
        .args(args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let token = env::var("HF_TOKEN").ok();
    let output = spawn_curl(command, token.as_deref())?.wait_with_output()?;
    if !output.status.success() {
        return Err(AppError::Message(format!(
            "Hugging Face request failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        )));
    }
    Ok(output.stdout)
}

fn authenticated_redirect_url(url: &str, token: &str) -> Result<Option<String>> {
    let curl = find_executable("curl")
        .ok_or_else(|| AppError::Message("Install curl to download models.".into()))?;
    let mut command = Command::new(curl);
    command
        .args([
            "--fail",
            "--silent",
            "--show-error",
            "--head",
            "--dump-header",
            "-",
        ])
        .arg(url)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let output = spawn_curl(command, Some(token))?.wait_with_output()?;
    if !output.status.success() {
        return Err(AppError::Message(format!(
            "Could not authorize access to the model file (HTTP/transfer status {}). Check HF_TOKEN and repository access.",
            output.status
        )));
    }
    let headers = String::from_utf8_lossy(&output.stdout);
    Ok(headers.lines().find_map(|line| {
        line.split_once(':')
            .filter(|(name, _)| name.eq_ignore_ascii_case("location"))
            .map(|(_, value)| value.trim().to_owned())
    }))
}

fn spawn_curl(mut command: Command, token: Option<&str>) -> Result<std::process::Child> {
    command.env_remove("HF_TOKEN");
    if let Some(token) = token {
        if !token.chars().all(|character| {
            character.is_ascii_alphanumeric() || matches!(character, '-' | '_' | '.')
        }) {
            return Err(AppError::Message(
                "HF_TOKEN contains unsupported characters.".into(),
            ));
        }
        command.arg("--config").arg("-").stdin(Stdio::piped());
    }
    let mut child = command.spawn()?;
    if let Some(token) = token {
        let config = format!("header = \"{} {} {}\"\n", "Authorization:", "Bearer", token);
        child
            .stdin
            .take()
            .ok_or_else(|| AppError::Message("Could not configure authenticated request.".into()))?
            .write_all(config.as_bytes())?;
    }
    Ok(child)
}

fn safe_relative_path(path: &str) -> bool {
    let path = Path::new(path);
    !path.as_os_str().is_empty()
        && path
            .components()
            .all(|component| matches!(component, Component::Normal(_)))
}

fn safe_repo_name(id: &str) -> String {
    id.split('/')
        .map(|segment| {
            segment
                .chars()
                .map(|character| {
                    if character.is_ascii_alphanumeric() || matches!(character, '-' | '_' | '.') {
                        character
                    } else {
                        '-'
                    }
                })
                .collect::<String>()
        })
        .collect::<Vec<_>>()
        .join("--")
}

fn encode_repo(id: &str) -> String {
    id.split('/')
        .map(encode_query)
        .collect::<Vec<_>>()
        .join("/")
}

fn encode_query(value: &str) -> String {
    value
        .bytes()
        .map(|byte| {
            if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b'~') {
                (byte as char).to_string()
            } else {
                format!("%{byte:02X}")
            }
        })
        .collect()
}

fn find_executable(name: &str) -> Option<PathBuf> {
    env::var_os("PATH")
        .into_iter()
        .flat_map(|paths| env::split_paths(&paths).collect::<Vec<_>>())
        .map(|directory| directory.join(name))
        .find(|path| path.is_file())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_metadata_uses_hub_card_fields() {
        let model: ApiModel = serde_json::from_str(
            r#"{
                "id":"author/model",
                "tags":["text-to-speech","expressive"],
                "cardData":{"language":["en","fr"],"license":"apache-2.0"},
                "config":{"architectures":["SpeechModel"]}
            }"#,
        )
        .expect("Hub model metadata parses");
        assert_eq!(model_languages(&model), ["en", "fr"]);
        assert_eq!(model_license(&model), "apache-2.0");
        assert_eq!(model_architecture(&model), "SpeechModel");
        assert!(model_tags(&model).contains(&"expressive".to_owned()));
    }

    #[test]
    fn filesystem_file_paths_must_be_relative_and_normal() {
        assert!(safe_relative_path("weights/model.safetensors"));
        assert!(!safe_relative_path("../outside"));
        assert!(!safe_relative_path("/absolute/path"));
    }

    #[test]
    fn byte_formatting_uses_binary_units() {
        assert_eq!(format_bytes(1024 * 1024), "1.0 MB");
    }
}
