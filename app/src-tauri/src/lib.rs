use base64::{Engine as _, engine::general_purpose::STANDARD};
use serde::Serialize;
use std::{
    fs,
    io::{Read, Write},
    path::{Path, PathBuf},
    process::{Command, Stdio},
    thread,
};
use tauri::{AppHandle, Emitter};

#[derive(Serialize)]
struct ImagePayload {
    path: String,
    name: String,
    data_url: String,
    bytes: u64,
}

#[derive(Serialize)]
struct EngineStatus {
    available: bool,
    detail: String,
}

#[derive(Clone, Serialize)]
struct RepairProgress {
    line: String,
    stream: &'static str,
}

fn relay_output<R: Read + Send + 'static>(
    mut reader: R,
    app: AppHandle,
    stream: &'static str,
) -> thread::JoinHandle<String> {
    thread::spawn(move || {
        let mut chunk = [0_u8; 4096];
        let mut current_line = Vec::new();
        let mut recent_lines = Vec::new();
        let publish = |bytes: &mut Vec<u8>, recent: &mut Vec<String>| {
            let line = String::from_utf8_lossy(bytes).trim().chars().take(2000).collect::<String>();
            bytes.clear();
            if line.is_empty() { return; }
            let _ = app.emit("repair-progress", RepairProgress { line: line.clone(), stream });
            recent.push(line);
            if recent.len() > 40 { recent.remove(0); }
        };
        loop {
            match reader.read(&mut chunk) {
                Ok(0) => break,
                Ok(count) => {
                    for byte in &chunk[..count] {
                        if *byte == b'\n' || *byte == b'\r' {
                            publish(&mut current_line, &mut recent_lines);
                        } else if current_line.len() < 8192 {
                            current_line.push(*byte);
                        }
                    }
                }
                Err(error) => {
                    recent_lines.push(format!("Could not read process output: {error}"));
                    break;
                }
            }
        }
        publish(&mut current_line, &mut recent_lines);
        recent_lines.join("\n")
    })
}

fn image_mime(path: &Path) -> Result<&'static str, String> {
    match path.extension().and_then(|x| x.to_str()).unwrap_or("").to_ascii_lowercase().as_str() {
        "png" => Ok("image/png"),
        "jpg" | "jpeg" => Ok("image/jpeg"),
        "webp" => Ok("image/webp"),
        _ => Err("Choose a PNG, JPEG, or WebP image.".into()),
    }
}

fn research_dir() -> Option<PathBuf> {
    let executable = std::env::current_exe().ok()?;
    executable.ancestors()
        .map(|parent| parent.join("research"))
        .find(|folder| folder.join("Remove-With-Flux2.ps1").is_file())
}

#[tauri::command]
fn read_image(path: String, prefer_original: Option<bool>) -> Result<ImagePayload, String> {
    let mut path = PathBuf::from(path);
    if prefer_original.unwrap_or(false) {
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
        if name.ends_with("_text_highlighted.png") {
            let manifest = path.with_extension("json");
            let content = fs::read_to_string(&manifest)
                .map_err(|_| "This is a detection preview. Select the unmarked source image instead.".to_string())?;
            let data: serde_json::Value = serde_json::from_str(&content).map_err(|e| e.to_string())?;
            let source = data.get("source").and_then(|v| v.as_str())
                .ok_or("Detection preview has no source path.")?;
            path = PathBuf::from(source);
        }
    }
    let mime = image_mime(&path)?;
    let metadata = fs::metadata(&path).map_err(|e| e.to_string())?;
    if !metadata.is_file() { return Err("The selected path is not a file.".into()); }
    if metadata.len() > 100 * 1024 * 1024 { return Err("Images over 100 MB are not supported in this preview.".into()); }
    let bytes = fs::read(&path).map_err(|e| e.to_string())?;
    Ok(ImagePayload {
        path: path.to_string_lossy().into_owned(),
        name: path.file_name().unwrap_or_default().to_string_lossy().into_owned(),
        data_url: format!("data:{mime};base64,{}", STANDARD.encode(bytes)),
        bytes: metadata.len(),
    })
}

#[tauri::command]
fn engine_status() -> EngineStatus {
    if !cfg!(target_os = "windows") {
        return EngineStatus { available: false, detail: "The research workflow currently runs on Windows only.".into() };
    }
    let Some(folder) = research_dir() else {
        return EngineStatus { available: false, detail: "Research folder not found next to this project.".into() };
    };
    for required in [
        "gpu_venv/Scripts/python.exe",
        "flux_compat_venv/Scripts/python.exe",
        "models/flux2-klein-4b-sdnq/model_index.json",
    ] {
        if !folder.join(required).is_file() {
            return EngineStatus { available: false, detail: format!("Research engine missing: {required}") };
        }
    }
    EngineStatus { available: true, detail: "Local research engine ready".into() }
}

#[tauri::command]
fn text_engine_status() -> EngineStatus {
    if !cfg!(target_os = "windows") {
        return EngineStatus { available: false, detail: "The local text workflow currently runs on Windows only.".into() };
    }
    let Some(folder) = research_dir() else {
        return EngineStatus { available: false, detail: "Research folder not found next to this project.".into() };
    };
    for required in [
        "Remove-Card-Text.ps1",
        "highlight_card_text.py",
        "probe_text_flux2_tiled.py",
        "text_detect_venv/Scripts/python.exe",
        "flux_compat_venv/Scripts/python.exe",
        "models/flux2-klein-4b-sdnq/model_index.json",
    ] {
        if !folder.join(required).is_file() {
            return EngineStatus { available: false, detail: format!("Text engine missing: {required}") };
        }
    }
    EngineStatus { available: true, detail: "Local text engine ready".into() }
}

#[tauri::command]
fn effect_engine_status() -> EngineStatus {
    if !cfg!(target_os = "windows") {
        return EngineStatus { available: false, detail: "Effect extraction currently runs on Windows only.".into() };
    }
    let Some(folder) = research_dir() else {
        return EngineStatus { available: false, detail: "Research folder not found next to this project.".into() };
    };
    for required in [
        "Extract-Card-Effects.ps1",
        "extract_effect_text.py",
        "highlight_card_text.py",
        "text_detect_venv/Scripts/python.exe",
    ] {
        if !folder.join(required).is_file() {
            return EngineStatus { available: false, detail: format!("Effect extractor missing: {required}") };
        }
    }
    EngineStatus { available: true, detail: "Local effect extractor ready".into() }
}

#[tauri::command]
fn translation_engine_status() -> EngineStatus {
    if !cfg!(target_os = "windows") {
        return EngineStatus { available: false, detail: "Offline translation currently runs on Windows only.".into() };
    }
    let Some(folder) = research_dir() else {
        return EngineStatus { available: false, detail: "Research folder not found next to this project.".into() };
    };
    for required in [
        "translate_effect_qwen.py",
        "card_rule_parser.py",
        "flux_compat_venv/Scripts/python.exe",
        "models/qwen3-1.7b/config.json",
        "models/qwen3-1.7b/model.safetensors.index.json",
    ] {
        if !folder.join(required).is_file() {
            return EngineStatus { available: false, detail: format!("Translator missing: {required}") };
        }
    }
    EngineStatus { available: true, detail: "Offline English translator ready".into() }
}

#[tauri::command]
async fn translate_card_effects(
    app: AppHandle, draft_path: String, source_text: String, language: String,
) -> Result<serde_json::Value, String> {
    tauri::async_runtime::spawn_blocking(move || {
        let status = translation_engine_status();
        if !status.available { return Err(status.detail); }
        if language != "ja" && language != "fr" {
            return Err("Choose Japanese or French as the source language.".into());
        }
        if source_text.trim().is_empty() || source_text.len() > 20_000 {
            return Err("The source effect must contain 1 to 20,000 characters.".into());
        }
        let draft = PathBuf::from(draft_path);
        if !draft.is_absolute() || !draft.is_file() || draft.extension().and_then(|x| x.to_str()) != Some("json") {
            return Err("Extract card effects before translating.".into());
        }
        let content = fs::read_to_string(&draft).map_err(|e| e.to_string())?;
        let data: serde_json::Value = serde_json::from_str(&content).map_err(|e| e.to_string())?;
        let source = data.get("source").and_then(|v| v.as_str()).ok_or("Effect draft has no source image.")?;
        let image = PathBuf::from(source);
        let parent = image.parent().ok_or("Source image has no parent folder.")?;
        let stem = image.file_stem().and_then(|s| s.to_str()).ok_or("Source image has no name.")?;
        let output = (1..1000).map(|index| {
            let suffix = if index == 1 { String::new() } else { format!("_{index}") };
            parent.join(format!("{stem}_effects_english{suffix}.json"))
        }).find(|path| !path.exists()).ok_or("Too many prior translations.")?;
        let folder = research_dir().ok_or("Research folder not found.")?;
        let mut child = Command::new(folder.join("flux_compat_venv/Scripts/python.exe"))
            .arg(folder.join("translate_effect_qwen.py"))
            .arg(&draft).arg("--language").arg(&language)
            .arg("--output").arg(&output).arg("--text-stdin")
            .env("PYTHONIOENCODING", "utf-8")
            .stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped())
            .spawn().map_err(|e| e.to_string())?;
        child.stdin.take().ok_or("Could not send edited OCR text.")?
            .write_all(source_text.as_bytes()).map_err(|e| e.to_string())?;
        let stdout = child.stdout.take().ok_or("Could not capture translation output.")?;
        let stderr = child.stderr.take().ok_or("Could not capture translation errors.")?;
        let output_thread = relay_output(stdout, app.clone(), "stdout");
        let error_thread = relay_output(stderr, app, "stderr");
        let status = child.wait().map_err(|e| e.to_string())?;
        let log = output_thread.join().unwrap_or_default();
        let error = error_thread.join().unwrap_or_default();
        if !status.success() {
            return Err(format!("Translation failed.\n{}\n{}", error.trim(), log.trim()));
        }
        let content = fs::read_to_string(&output).map_err(|e| e.to_string())?;
        let mut result: serde_json::Value = serde_json::from_str(&content).map_err(|e| e.to_string())?;
        result["output_path"] = output.to_string_lossy().into_owned().into();
        Ok(result)
    }).await.map_err(|e| e.to_string())?
}

fn next_output_path(input: &Path, label: &str) -> Result<PathBuf, String> {
    let parent = input.parent().ok_or("Input has no parent folder")?;
    let stem = input.file_stem().and_then(|s| s.to_str()).ok_or("Input has no file name")?;
    for index in 1..1000 {
        let suffix = if index == 1 { String::new() } else { format!("_{index}") };
        let candidate = parent.join(format!("{stem}_{label}{suffix}.png"));
        if !candidate.exists() { return Ok(candidate); }
    }
    Err("Too many prior outputs with this name.".into())
}

#[tauri::command]
async fn extract_card_effects(app: AppHandle, image_path: String) -> Result<serde_json::Value, String> {
    tauri::async_runtime::spawn_blocking(move || {
        let input = PathBuf::from(image_path);
        if !input.is_absolute() || !input.is_file() {
            return Err("Choose an existing image with an absolute path.".into());
        }
        image_mime(&input)?;
        let folder = research_dir().ok_or("Research folder not found.")?;
        let script = folder.join("Extract-Card-Effects.ps1");
        if !script.is_file() { return Err("Effect extraction script is missing.".into()); }
        let parent = input.parent().ok_or("Input has no parent folder.")?;
        let stem = input.file_stem().and_then(|s| s.to_str()).ok_or("Input has no file name.")?;
        let output = (1..1000).map(|index| {
            let suffix = if index == 1 { String::new() } else { format!("_{index}") };
            parent.join(format!("{stem}_effects{suffix}.json"))
        }).find(|path| !path.exists()).ok_or("Too many prior effect drafts.")?;
        let mut child = Command::new("powershell.exe")
            .args(["-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File"])
            .arg(script).arg("-Image").arg(&input).arg("-Output").arg(&output)
            .stdout(Stdio::piped()).stderr(Stdio::piped())
            .spawn().map_err(|e| e.to_string())?;
        let stdout = child.stdout.take().ok_or("Could not capture extraction output.")?;
        let stderr = child.stderr.take().ok_or("Could not capture extraction errors.")?;
        let output_thread = relay_output(stdout, app.clone(), "stdout");
        let error_thread = relay_output(stderr, app, "stderr");
        let status = child.wait().map_err(|e| e.to_string())?;
        let log = output_thread.join().unwrap_or_default();
        let error = error_thread.join().unwrap_or_default();
        if !status.success() {
            return Err(format!("Effect extraction failed.\n{}\n{}", error.trim(), log.trim()));
        }
        let content = fs::read_to_string(&output).map_err(|e| e.to_string())?;
        let mut data: serde_json::Value = serde_json::from_str(&content).map_err(|e| e.to_string())?;
        data["output_path"] = output.to_string_lossy().into_owned().into();
        Ok(data)
    }).await.map_err(|e| e.to_string())?
}

#[tauri::command]
async fn run_research(app: AppHandle, image_path: String, seed: i32) -> Result<String, String> {
    tauri::async_runtime::spawn_blocking(move || {
        let status = engine_status();
        if !status.available { return Err(status.detail); }
        let input = PathBuf::from(image_path);
        if !input.is_absolute() || !input.is_file() {
            return Err("Choose an existing image with an absolute path.".into());
        }
        image_mime(&input)?;
        let output = next_output_path(&input, "flux2_klein_2x")?;
        let script = research_dir().ok_or("Research folder not found.")?.join("Remove-With-Flux2.ps1");
        let mut child = Command::new("powershell.exe")
            .args(["-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File"])
            .arg(script)
            .arg("-Image").arg(&input)
            .arg("-Output").arg(&output)
            .arg("-Seed").arg(seed.to_string())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn().map_err(|e| e.to_string())?;
        let stdout = child.stdout.take().ok_or("Could not capture workflow output.")?;
        let stderr = child.stderr.take().ok_or("Could not capture workflow errors.")?;
        let output_thread = relay_output(stdout, app.clone(), "stdout");
        let error_thread = relay_output(stderr, app, "stderr");
        let status = child.wait().map_err(|e| e.to_string())?;
        let log = output_thread.join().unwrap_or_default();
        let error = error_thread.join().unwrap_or_default();
        if !status.success() {
            return Err(format!("Removal failed.\n{}\n{}", error.trim(), log.trim()));
        }
        if !output.is_file() { return Err("The research engine returned without an output image.".into()); }
        Ok(output.to_string_lossy().into_owned())
    }).await.map_err(|e| e.to_string())?
}

#[tauri::command]
async fn run_text_research(app: AppHandle, image_path: String, include_vertical: bool) -> Result<String, String> {
    tauri::async_runtime::spawn_blocking(move || {
        let status = text_engine_status();
        if !status.available { return Err(status.detail); }
        let input = PathBuf::from(image_path);
        if input.file_name().and_then(|n| n.to_str()).unwrap_or("").ends_with("_text_highlighted.png") {
            return Err("Select the unmarked source image, not a red-box detection preview.".into());
        }
        if !input.is_absolute() || !input.is_file() {
            return Err("Choose an existing image with an absolute path.".into());
        }
        image_mime(&input)?;
        let output = next_output_path(&input, "text_removed")?;
        let script = research_dir().ok_or("Research folder not found.")?.join("Remove-Card-Text.ps1");
        let mut command = Command::new("powershell.exe");
        command.args(["-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File"])
            .arg(script).arg("-Image").arg(&input).arg("-Output").arg(&output);
        if include_vertical { command.arg("-IncludeVertical"); }
        let mut child = command
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn().map_err(|e| e.to_string())?;
        let stdout = child.stdout.take().ok_or("Could not capture workflow output.")?;
        let stderr = child.stderr.take().ok_or("Could not capture workflow errors.")?;
        let output_thread = relay_output(stdout, app.clone(), "stdout");
        let error_thread = relay_output(stderr, app, "stderr");
        let status = child.wait().map_err(|e| e.to_string())?;
        let log = output_thread.join().unwrap_or_default();
        let error = error_thread.join().unwrap_or_default();
        if !status.success() {
            return Err(format!("Text removal failed.\n{}\n{}", error.trim(), log.trim()));
        }
        if !output.is_file() { return Err("The text engine returned without an output image.".into()); }
        Ok(output.to_string_lossy().into_owned())
    }).await.map_err(|e| e.to_string())?
}

#[tauri::command]
fn export_image(source_path: String, destination_path: String) -> Result<(), String> {
    let source = PathBuf::from(source_path).canonicalize().map_err(|e| e.to_string())?;
    image_mime(&source)?;
    let destination = PathBuf::from(destination_path);
    if destination.exists() && destination.canonicalize().map_err(|e| e.to_string())? == source {
        return Err("Choose a different file name for the export.".into());
    }
    fs::copy(source, destination).map_err(|e| e.to_string())?;
    Ok(())
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .invoke_handler(tauri::generate_handler![read_image, engine_status, text_engine_status, effect_engine_status, translation_engine_status, run_research, run_text_research, extract_card_effects, translate_card_effects, export_image])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
