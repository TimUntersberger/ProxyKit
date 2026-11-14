import { invoke, isTauri } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { open, save } from "@tauri-apps/plugin-dialog";
import { initCardGenerator } from "./card-generator";

type ImagePayload = { path: string; name: string; data_url: string; bytes: number };
type EngineStatus = { available: boolean; detail: string };
type RepairProgress = { line: string; stream: string };
type EffectDraft = { draft_text: string; output_path: string; lines: { text: string; confidence: number; role: string }[] };
type TranslationDraft = { translation: string; output_path: string; warnings: string[] };
type View = "before" | "compare" | "after";

const el = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
let original: ImagePayload | null = null;
let result: ImagePayload | null = null;
let sourceDimensions: { width: number; height: number } | null = null;
let resultDimensions: { width: number; height: number } | null = null;
let engine: EngineStatus = { available: false, detail: "Checking local engine…" };
let textEngine: EngineStatus = { available: false, detail: "Checking text engine…" };
let effectEngine: EngineStatus = { available: false, detail: "Checking effect extractor…" };
let translationEngine: EngineStatus = { available: false, detail: "Checking offline translator…" };
let effectDraftPath: string | null = null;
let activeWorkflow: "watermark" | "text" | "extract" | "translate" = "watermark";
let busy = false;
let view: View = "compare";
let zoom = 1;
let panX = 0;
let panY = 0;
let draggingHandle = false;
let draggingPan = false;
let pointerX = 0;
let pointerY = 0;
let progressState: "idle" | "running" | "done" | "error" = "idle";
let progressStarted = 0;
let lastActivity = 0;
let progressLines: string[] = [];

function getDimensions(dataUrl: string): Promise<{ width: number; height: number }> {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve({ width: image.naturalWidth, height: image.naturalHeight });
    image.onerror = () => reject(new Error("This image could not be decoded."));
    image.src = dataUrl;
  });
}

function message(text: string, error = false) {
  const node = el<HTMLElement>("notice");
  node.textContent = text;
  node.classList.toggle("error", error);
  node.classList.remove("hidden");
}

function renderProgress() {
  const card = el<HTMLElement>("progress-card");
  card.classList.toggle("hidden", progressState === "idle");
  card.classList.toggle("complete", progressState === "done");
  card.classList.toggle("failed", progressState === "error");
  el<HTMLElement>("activity-bar").classList.toggle("stopped", progressState !== "running");
  if (progressStarted) {
    const seconds = Math.floor((Date.now() - progressStarted) / 1000);
    el<HTMLElement>("progress-elapsed").textContent = `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, "0")}`;
  }
  if (progressState === "running" && Date.now() - lastActivity > 20000) {
    el<HTMLElement>("progress-detail").textContent = activeWorkflow === "extract"
      ? "Still detecting text locally…"
      : activeWorkflow === "translate" ? "Still translating locally…"
      : "Still working locally. Model loading and repainting can take several minutes.";
  }
}

function addProgressLine(payload: RepairProgress) {
  const line = payload.line.replace(/\x1b\[[0-9;]*m/g, "").trim();
  if (!line) return;
  lastActivity = Date.now();
  progressLines.push(line);
  if (progressLines.length > 12) progressLines.shift();
  el<HTMLElement>("progress-log").textContent = progressLines.join("\n");
  el<HTMLElement>("progress-detail").textContent = line;
  if (line.includes("Step 1/2")) el<HTMLElement>("progress-step").textContent = activeWorkflow === "text" ? "Step 1 of 2 · Detecting text" : "Step 1 of 2 · Source repair";
  else if (line.includes("Step 2/2")) el<HTMLElement>("progress-step").textContent = "Step 2 of 2 · Model repaint";
  else if (line.includes("Loading Klein")) el<HTMLElement>("progress-step").textContent = "Step 2 of 2 · Loading model";
  else if (line.includes("Upscaling")) el<HTMLElement>("progress-step").textContent = "Step 1 of 2 · Upscaling";
}

function applyViewTransform() {
  el<HTMLElement>("image-stage").style.transform =
    `translate(-50%, -50%) translate(${panX}px, ${panY}px) scale(${zoom})`;
  el<HTMLElement>("zoom-label").textContent = `${Math.round(zoom * 100)}%`;
}

function fitStage() {
  const dimensions = original ? sourceDimensions : resultDimensions;
  if (!dimensions) return;
  const host = el<HTMLElement>("preview-stage");
  const fit = Math.min((host.clientWidth - 36) / dimensions.width, (host.clientHeight - 36) / dimensions.height);
  if (!Number.isFinite(fit) || fit <= 0) return;
  const stage = el<HTMLElement>("image-stage");
  stage.style.width = `${Math.floor(dimensions.width * fit)}px`;
  stage.style.height = `${Math.floor(dimensions.height * fit)}px`;
  applyViewTransform();
}

function setZoom(next: number) {
  zoom = Math.max(0.5, Math.min(6, next));
  if (zoom <= 1) { panX = 0; panY = 0; }
  applyViewTransform();
}

function render() {
  el<HTMLElement>("engine-status").textContent = `Watermark: ${engine.detail} · Text: ${textEngine.detail} · OCR: ${effectEngine.detail} · Translation: ${translationEngine.detail}`;
  el<HTMLElement>("engine-dot").classList.toggle("ready", engine.available || textEngine.available || effectEngine.available || translationEngine.available);
  el<HTMLElement>("source-file").classList.toggle("hidden", !original);
  el<HTMLElement>("open-original").classList.toggle("hidden", !!original);
  const supportedSize = !!sourceDimensions && sourceDimensions.width <= 640 && sourceDimensions.height >= 512;
  el<HTMLButtonElement>("run-button").disabled = !original || !engine.available || !supportedSize || busy;
  document.querySelectorAll<HTMLInputElement>('input[name="watermark-version"]').forEach(input => { input.disabled = busy; });
  const textSupported = !!sourceDimensions;
  el<HTMLButtonElement>("text-run-button").disabled = !original || !textEngine.available || !textSupported || busy;
  el<HTMLButtonElement>("extract-effects").disabled = !original || !effectEngine.available || busy;
  el<HTMLButtonElement>("translate-effects").disabled = !effectDraftPath || !translationEngine.available || busy;
  el<HTMLSelectElement>("effect-language").disabled = busy;
  el<HTMLElement>("text-run-help").textContent = !textEngine.available ? textEngine.detail
    : "Detects high-confidence text; saves a new PNG beside the source.";
  el<HTMLButtonElement>("export-button").disabled = !result || busy;
  el<HTMLButtonElement>("open-result").disabled = busy;
  el<HTMLButtonElement>("change-original").disabled = busy;
  el<HTMLElement>("run-label").textContent = busy && activeWorkflow === "watermark" ? "Repairing image…" : "Remove watermark";
  el<HTMLElement>("run-help").textContent = busy
    ? "This can take several minutes. The output will be saved next to the source image."
    : !engine.available ? engine.detail
    : original && !supportedSize ? "This research model needs a source at most 640 px wide and at least 512 px high."
    : "Output is saved next to the source image.";
  el<HTMLElement>("result-path").textContent = result ? result.path : "";
  if (original) {
    el<HTMLElement>("source-name").textContent = original.name;
    const size = sourceDimensions ? `${sourceDimensions.width} × ${sourceDimensions.height} · ` : "";
    el<HTMLElement>("source-meta").textContent = `${size}${(original.bytes / 1024 / 1024).toFixed(2)} MB · ${original.path}`;
    el<HTMLImageElement>("before-image").src = original.data_url;
  }
  el<HTMLElement>("empty-state").classList.toggle("hidden", !!original || !!result);
  el<HTMLElement>("image-stage").classList.toggle("hidden", !original && !result);
  el<HTMLElement>("preview-tools").classList.toggle("hidden", !original && !result);
  const hasPair = !!original && !!result;
  el<HTMLElement>("view-toggle").classList.toggle("hidden", !hasPair);
  el<HTMLElement>("compare-controls").classList.toggle("hidden", !hasPair || view !== "compare");
  el<HTMLElement>("preview-subtitle").textContent = hasPair ? "Drag the divider to inspect the repair"
    : original ? "Original image" : result ? "Repaired image" : "Your image will appear here";
  if (result) el<HTMLImageElement>("after-image").src = result.data_url;
  const before = el<HTMLImageElement>("before-image");
  const after = el<HTMLImageElement>("after-image");
  before.classList.toggle("hidden", !original || (hasPair && view === "after"));
  after.classList.toggle("hidden", !result || (hasPair && view === "before"));
  after.classList.toggle("compare-image", hasPair && view === "compare");
  el<HTMLElement>("split-line").classList.toggle("hidden", !hasPair || view !== "compare");
  document.querySelectorAll<HTMLButtonElement>("[data-view]").forEach(button => {
    button.classList.toggle("selected", button.dataset.view === view);
  });
  updateSplit();
  fitStage();
  renderProgress();
}

function updateSplit() {
  const value = el<HTMLInputElement>("compare-slider").value;
  el<HTMLElement>("image-stage").style.setProperty("--split", `${value}%`);
  el<HTMLElement>("split-line").style.left = `${value}%`;
  el<HTMLElement>("split-line").setAttribute("aria-valuenow", value);
}

function splitAtPointer(clientX: number) {
  const bounds = el<HTMLElement>("image-stage").getBoundingClientRect();
  if (!bounds.width) return;
  el<HTMLInputElement>("compare-slider").value =
    String(Math.round(Math.max(0, Math.min(100, (clientX - bounds.left) / bounds.width * 100))));
  updateSplit();
}

async function chooseOriginal() {
  try {
    const path = await open({ multiple: false, directory: false, filters: [{ name: "Images", extensions: ["png", "jpg", "jpeg", "webp"] }] });
    if (typeof path !== "string") return;
    const image = await invoke<ImagePayload>("read_image", { path, preferOriginal: true });
    const dimensions = await getDimensions(image.data_url);
    original = image;
    sourceDimensions = dimensions;
    result = null;
    resultDimensions = null;
    view = "compare";
    zoom = 1;
    panX = 0;
    panY = 0;
    progressState = "idle";
    effectDraftPath = null;
    el<HTMLElement>("effect-draft").classList.add("hidden");
    el<HTMLElement>("translation-draft").classList.add("hidden");
    el<HTMLElement>("notice").classList.add("hidden");
    render();
  } catch (error) { message(String(error), true); }
}

async function chooseResult() {
  try {
    const path = await open({ multiple: false, directory: false, filters: [{ name: "Images", extensions: ["png", "jpg", "jpeg", "webp"] }] });
    if (typeof path !== "string") return;
    const image = await invoke<ImagePayload>("read_image", { path });
    const dimensions = await getDimensions(image.data_url);
    result = image;
    resultDimensions = dimensions;
    view = original ? "compare" : "after";
    setZoom(1);
    render();
  } catch (error) { message(String(error), true); }
}

async function runResearch(workflow: "watermark" | "text") {
  if (!original || busy) return;
  const version = document.querySelector<HTMLInputElement>('input[name="watermark-version"]:checked')?.value;
  const seed = version === "alternate" ? 777 : 101;
  activeWorkflow = workflow;
  busy = true;
  progressState = "running";
  progressStarted = Date.now();
  lastActivity = progressStarted;
  progressLines = [];
  el<HTMLElement>("progress-step").textContent = "Starting local workflow";
  el<HTMLElement>("progress-detail").textContent = workflow === "text" ? "Preparing text detection…" : "Preparing source image…";
  el<HTMLElement>("progress-log").textContent = "";
  el<HTMLElement>("notice").classList.add("hidden");
  render();
  let stopListening: (() => void) | undefined;
  const timer = window.setInterval(renderProgress, 1000);
  try {
    stopListening = await listen<RepairProgress>("repair-progress", event => addProgressLine(event.payload));
    const path = await invoke<string>(workflow === "text" ? "run_text_research" : "run_research", {
      imagePath: original.path,
      seed,
      includeVertical: workflow === "text" && el<HTMLInputElement>("include-vertical").checked,
    });
    const image = await invoke<ImagePayload>("read_image", { path });
    result = image;
    resultDimensions = await getDimensions(image.data_url);
    view = "compare";
    progressState = "done";
    el<HTMLElement>("progress-step").textContent = "Repair complete";
    el<HTMLElement>("progress-detail").textContent = "The result is ready to inspect.";
    message(workflow === "text" ? "Text-removal result ready. Inspect the reconstructed artwork before using it." : "Repair complete. Compare the result, then save a copy if you like.");
  } catch (error) {
    progressState = "error";
    el<HTMLElement>("progress-step").textContent = "Repair stopped";
    el<HTMLElement>("progress-detail").textContent = "See the activity log and error below.";
    message(String(error), true);
  } finally {
    window.clearInterval(timer);
    stopListening?.();
    busy = false;
    render();
  }
}

async function extractEffects() {
  if (!original || busy) return;
  activeWorkflow = "extract";
  busy = true;
  progressState = "running";
  progressStarted = Date.now();
  lastActivity = progressStarted;
  progressLines = [];
  el<HTMLElement>("progress-step").textContent = "Extracting card effects";
  el<HTMLElement>("progress-detail").textContent = "Detecting printed text locally…";
  el<HTMLElement>("progress-log").textContent = "";
  el<HTMLElement>("notice").classList.add("hidden");
  render();
  let stopListening: (() => void) | undefined;
  const timer = window.setInterval(renderProgress, 1000);
  try {
    stopListening = await listen<RepairProgress>("repair-progress", event => addProgressLine(event.payload));
    const draft = await invoke<EffectDraft>("extract_card_effects", { imagePath: original.path });
    effectDraftPath = draft.output_path;
    el<HTMLTextAreaElement>("effect-text").value = draft.draft_text;
    el<HTMLElement>("translation-draft").classList.add("hidden");
    el<HTMLSelectElement>("effect-language").value = /[\u3040-\u30ff\u3400-\u9fff]/.test(draft.draft_text) ? "ja" : "fr";
    el<HTMLElement>("effect-path").textContent = `OCR details: ${draft.output_path}`;
    el<HTMLElement>("effect-draft").classList.remove("hidden");
    progressState = "done";
    el<HTMLElement>("progress-step").textContent = "Effect draft ready";
    el<HTMLElement>("progress-detail").textContent = "Review the text against the original card.";
    const uncertain = draft.lines.filter(line => line.role === "uncertain").length;
    message(`Extracted ${draft.lines.filter(line => line.role === "effect").length} text regions.${uncertain ? ` ${uncertain} isolated number(s) need checking in the OCR details.` : ""} Check symbols, spaces, and line breaks before using the draft.`);
  } catch (error) {
    progressState = "error";
    el<HTMLElement>("progress-step").textContent = "Extraction stopped";
    el<HTMLElement>("progress-detail").textContent = "See the activity log and error below.";
    message(String(error), true);
  } finally {
    window.clearInterval(timer);
    stopListening?.();
    busy = false;
    render();
  }
}

async function translateEffects() {
  if (!effectDraftPath || busy) return;
  activeWorkflow = "translate";
  busy = true;
  progressState = "running";
  progressStarted = Date.now();
  lastActivity = progressStarted;
  progressLines = [];
  el<HTMLElement>("progress-step").textContent = "Translating card effects";
  el<HTMLElement>("progress-detail").textContent = "Parsing card effects; loading the offline model for unfamiliar wording…";
  el<HTMLElement>("progress-log").textContent = "";
  el<HTMLElement>("notice").classList.add("hidden");
  render();
  let stopListening: (() => void) | undefined;
  const timer = window.setInterval(renderProgress, 1000);
  try {
    stopListening = await listen<RepairProgress>("repair-progress", event => addProgressLine(event.payload));
    const draft = await invoke<TranslationDraft>("translate_card_effects", {
      draftPath: effectDraftPath,
      sourceText: el<HTMLTextAreaElement>("effect-text").value,
      language: el<HTMLSelectElement>("effect-language").value,
    });
    el<HTMLTextAreaElement>("translation-text").value = draft.translation;
    el<HTMLElement>("translation-path").textContent = `Translation details: ${draft.output_path}`;
    const warnings = el<HTMLElement>("translation-warnings");
    warnings.textContent = draft.warnings.join("\n");
    warnings.classList.toggle("hidden", draft.warnings.length === 0);
    el<HTMLElement>("translation-draft").classList.remove("hidden");
    progressState = "done";
    el<HTMLElement>("progress-step").textContent = "Translation draft ready";
    el<HTMLElement>("progress-detail").textContent = "Compare the English draft with the original effect.";
    message(draft.warnings.length
      ? `Translation draft ready. ${draft.warnings.length} rule check(s) need attention.`
      : "Translation draft ready. Review card terms and conditions before using it.");
  } catch (error) {
    progressState = "error";
    el<HTMLElement>("progress-step").textContent = "Translation stopped";
    el<HTMLElement>("progress-detail").textContent = "See the activity log and error below.";
    message(String(error), true);
  } finally {
    window.clearInterval(timer);
    stopListening?.();
    busy = false;
    render();
  }
}

async function exportResult() {
  if (!result) return;
  try {
    const destinationPath = await save({ defaultPath: result.name, filters: [{ name: "Image", extensions: [result.name.split(".").pop() || "png"] }] });
    if (!destinationPath) return;
    await invoke("export_image", { sourcePath: result.path, destinationPath });
    message(`Saved to ${destinationPath}`);
  } catch (error) { message(String(error), true); }
}

el<HTMLButtonElement>("open-original").addEventListener("click", chooseOriginal);
el<HTMLButtonElement>("change-original").addEventListener("click", chooseOriginal);
el<HTMLButtonElement>("open-result").addEventListener("click", chooseResult);
el<HTMLButtonElement>("run-button").addEventListener("click", () => runResearch("watermark"));
el<HTMLButtonElement>("text-run-button").addEventListener("click", () => runResearch("text"));
el<HTMLButtonElement>("extract-effects").addEventListener("click", extractEffects);
el<HTMLButtonElement>("translate-effects").addEventListener("click", translateEffects);
el<HTMLButtonElement>("copy-effect-text").addEventListener("click", async () => {
  try {
    await navigator.clipboard.writeText(el<HTMLTextAreaElement>("effect-text").value);
    message("Copied the edited effect text.");
  } catch (error) { message(`Could not copy text: ${error}`, true); }
});
el<HTMLButtonElement>("copy-translation").addEventListener("click", async () => {
  try {
    await navigator.clipboard.writeText(el<HTMLTextAreaElement>("translation-text").value);
    message("Copied the edited English draft.");
  } catch (error) { message(`Could not copy text: ${error}`, true); }
});
el<HTMLButtonElement>("export-button").addEventListener("click", exportResult);
el<HTMLInputElement>("compare-slider").addEventListener("input", updateSplit);
el<HTMLButtonElement>("zoom-in").addEventListener("click", () => setZoom(zoom * 1.25));
el<HTMLButtonElement>("zoom-out").addEventListener("click", () => setZoom(zoom / 1.25));
el<HTMLButtonElement>("zoom-fit").addEventListener("click", () => { panX = 0; panY = 0; setZoom(1); });
const previewStage = el<HTMLElement>("preview-stage");
previewStage.addEventListener("wheel", event => {
  if (!original && !result) return;
  event.preventDefault();
  setZoom(zoom * (event.deltaY < 0 ? 1.15 : 1 / 1.15));
}, { passive: false });
const splitHandle = el<HTMLElement>("split-line");
splitHandle.addEventListener("pointerdown", event => {
  event.preventDefault();
  event.stopPropagation();
  draggingHandle = true;
  splitHandle.setPointerCapture(event.pointerId);
  splitAtPointer(event.clientX);
});
splitHandle.addEventListener("pointermove", event => {
  if (draggingHandle) splitAtPointer(event.clientX);
});
const endHandleDrag = () => { draggingHandle = false; };
splitHandle.addEventListener("pointerup", endHandleDrag);
splitHandle.addEventListener("pointercancel", endHandleDrag);
splitHandle.addEventListener("keydown", event => {
  if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
  event.preventDefault();
  const slider = el<HTMLInputElement>("compare-slider");
  slider.value = String(Math.max(0, Math.min(100,
    Number(slider.value) + (event.key === "ArrowRight" ? 1 : -1) * (event.shiftKey ? 10 : 2))));
  updateSplit();
});
previewStage.addEventListener("pointerdown", event => {
  if (zoom <= 1 || (!original && !result)) return;
  draggingPan = true;
  pointerX = event.clientX;
  pointerY = event.clientY;
  previewStage.setPointerCapture(event.pointerId);
  previewStage.classList.add("panning");
});
previewStage.addEventListener("pointermove", event => {
  if (!draggingPan) return;
  panX += event.clientX - pointerX;
  panY += event.clientY - pointerY;
  pointerX = event.clientX;
  pointerY = event.clientY;
  applyViewTransform();
});
const endPan = () => { draggingPan = false; previewStage.classList.remove("panning"); };
previewStage.addEventListener("pointerup", endPan);
previewStage.addEventListener("pointercancel", endPan);
new ResizeObserver(fitStage).observe(previewStage);
document.querySelectorAll<HTMLButtonElement>("[data-view]").forEach(button => {
  button.addEventListener("click", () => { view = button.dataset.view as View; render(); });
});

render();
initCardGenerator();
const showPage = (page: "repair" | "card") => {
  el<HTMLElement>("repair-page").classList.toggle("hidden", page !== "repair");
  el<HTMLElement>("card-page").classList.toggle("hidden", page !== "card");
  el<HTMLElement>("nav-repair").classList.toggle("active", page === "repair");
  el<HTMLElement>("nav-card").classList.toggle("active", page === "card");
  el<HTMLElement>("page-title").textContent = page === "card" ? "Card generator" : "Image repair";
  if (page === "repair") fitStage();
};
el<HTMLButtonElement>("nav-repair").addEventListener("click", () => showPage("repair"));
el<HTMLButtonElement>("nav-card").addEventListener("click", () => showPage("card"));
if (isTauri()) {
  invoke<EngineStatus>("engine_status").then(status => { engine = status; render(); })
    .catch(error => { engine = { available: false, detail: String(error) }; render(); });
  invoke<EngineStatus>("text_engine_status").then(status => { textEngine = status; render(); })
    .catch(error => { textEngine = { available: false, detail: String(error) }; render(); });
  invoke<EngineStatus>("effect_engine_status").then(status => { effectEngine = status; render(); })
    .catch(error => { effectEngine = { available: false, detail: String(error) }; render(); });
  invoke<EngineStatus>("translation_engine_status").then(status => { translationEngine = status; render(); })
    .catch(error => { translationEngine = { available: false, detail: String(error) }; render(); });
} else {
  engine = { available: false, detail: "Open with npm run tauri dev; the browser preview cannot run the engine." };
  textEngine = engine;
  effectEngine = engine;
  translationEngine = engine;
  render();
}
