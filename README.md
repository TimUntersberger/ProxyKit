# ProxyKit

A local desktop workspace for reviewing and removing translucent watermarks. The Tauri MVP lives in `app/`; the existing feasibility work and model files are preserved in `research/`.

## Project layout

| Path | Purpose |
| --- | --- |
| `app/` | Tauri 2 UI, TypeScript frontend, Rust file operations and workflow bridge |
| `research/` | Existing PowerShell/Python experiments, images, model files and environments |
| `research/REPORT.md` | Detailed research log and method limits |
| `research/Remove-With-Flux2.ps1` | Current experimental removal workflow |
| `research/Remove-Card-Text.ps1` | Experimental automatic printed-text removal workflow |

The root was reorganized without deleting the experiment or moving its scripts apart from their relative dependencies. Existing commands should now be run from `research/` or with an explicit `research/Remove-With-Flux2.ps1` path.

## Run the UI

```powershell
cd app
npm install
npm run tauri dev
```

For a full Windows processing setup on another PC, see [SETUP_WINDOWS.md](SETUP_WINDOWS.md).

The **Remove watermark** button currently uses the existing local research workflow when it is present next to the app on Windows. It writes a new `*_flux2_klein_2x.png` file beside the selected source image and never replaces an existing output. The UI can also open an existing result, compare it with the source, and save a copy.

The **Remove card text** button detects high-confidence text, repaints complex regions with the local FLUX.2 Klein research model, and writes a new `*_text_removed.png` beside the source. An optional checkbox includes vertical text. Plain white text areas are filled directly, and compact emblems are preserved when the underlying graphic is uncertain. It does not require a manual mask. The result is experimental: false detections can alter decoration, and missed text remains.

## Repository contents

This repository contains source code, scripts, tests, and research notes. Downloaded model weights, Python environments, sample card images, generated outputs, and local run folders are excluded. The Tauri UI builds without those files, but the Windows research actions need locally installed environments and models in `research/`. The app does not yet bundle those dependencies or provide a macOS processing backend.

This is a **UI MVP**, not the cross-platform Rust processing engine. The experimental engine remains Python-based and is not bundled into the desktop app. A packaged build or macOS/Linux build can review images but cannot run that engine yet. The next engineering step is to port or replace the repair pipeline behind the Rust command interface.
