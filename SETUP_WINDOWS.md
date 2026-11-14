# Set up ProxyKit on another Windows PC

The GitHub repository contains the app and research source, not the downloaded
models or Python environments. The processing workflows currently require
Windows and an NVIDIA GPU. macOS and Linux can build the UI, but the research
commands are disabled there.

## 1. Install the development tools

- Install Git, Node.js with npm, Rust, and Python 3.11 (the working environment
  used Python 3.11.9).
- Install the Microsoft C++ Build Tools and Edge WebView2 required by
  [Tauri on Windows](https://v2.tauri.app/start/prerequisites/).
- Install an NVIDIA driver compatible with the CUDA 12.1 PyTorch wheels. The
  [PyTorch 2.5.1 instructions](https://docs.pytorch.org/get-started/previous-versions/)
  list the CUDA 12.1 wheel index used by the current environment.
- Install [Upscayl](https://upscayl.org/) so the current scripts can find
  `C:\Program Files\Upscayl\resources\bin\upscayl-bin.exe` and its models at
  `C:\Program Files\Upscayl\resources\models`. The scripts use those exact paths.
- Check that `C:\Windows\Fonts\impact.ttf` exists. The SAMPLE detector and
  source reconstruction use that font by default.

## 2. Clone and build the UI

```powershell
git clone https://github.com/TimUntersberger/ProxyKit.git
cd ProxyKit\app
npm ci
npm run tauri dev
```

The UI can open and compare images at this point. Processing buttons become
available only when the research environments and model files below are present
in the sibling `research` folder.

## 3. Recreate the Python environments

Run these from the repository root in PowerShell. The three requirement files
are snapshots of the working Windows environments; recreate the environments
instead of copying the virtual-environment directories to a different path.

```powershell
py -3.11 -m venv research\gpu_venv
py -3.11 -m venv research\flux_compat_venv
py -3.11 -m venv research\text_detect_venv

.\research\gpu_venv\Scripts\python.exe -m pip install -r research\requirements-gpu-windows.txt --extra-index-url https://download.pytorch.org/whl/cu121
.\research\flux_compat_venv\Scripts\python.exe -m pip install -r research\requirements-flux-windows.txt --extra-index-url https://download.pytorch.org/whl/cu121
.\research\text_detect_venv\Scripts\python.exe -m pip install -r research\requirements-ocr-windows.txt
```

These snapshots record package versions, but a fresh installation has not yet
been tested. If a package is unavailable or the new GPU needs a different
PyTorch build, the environment may need an adjusted install command.

## 4. Restore local model files

- Copy the existing `research\models\flux2-klein-4b-sdnq\` folder from the
  working PC to the same relative location on the new PC. The current
  quantized FLUX.2 Klein pipeline expects its `model_index.json` and component
  weights there. This quantized model is not recreated by a script in the repo.
- Download the translation model with:

```powershell
.\research\flux_compat_venv\Scripts\python.exe .\research\setup_qwen_translation_model.py
```

This downloads `Qwen/Qwen3-1.7B` into `research\models\qwen3-1.7b\`.
Downloaded models and their licenses remain the responsibility of the local
installation; no model weights are committed to GitHub.

## 5. Check the installation

```powershell
.\research\gpu_venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available())"
.\research\flux_compat_venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available())"
cd app
npm run tauri dev
```

Both Python checks should print `True`. ProxyKit should report that its local
watermark, text, extraction, and translation engines are ready. The source
images and outputs are not in the repository; choose an image on the new PC.
