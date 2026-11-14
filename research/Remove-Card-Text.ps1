<#
Experimental printed-card-text removal. Detects high-confidence horizontal text
and repaints those regions with the local FLUX.2 Klein model. Outputs a new PNG.
#>
param(
    [Parameter(Mandatory = $true)] [string]$Image,
    [Parameter(Mandatory = $true)] [string]$Output,
    [switch]$IncludeVertical
)

$ErrorActionPreference = 'Stop'
$detectPython = Join-Path $PSScriptRoot 'text_detect_venv\Scripts\python.exe'
$fluxPython = Join-Path $PSScriptRoot 'flux_compat_venv\Scripts\python.exe'
$detectScript = Join-Path $PSScriptRoot 'highlight_card_text.py'
$repaintScript = Join-Path $PSScriptRoot 'probe_text_flux2_tiled.py'
$modelIndex = Join-Path $PSScriptRoot 'models\flux2-klein-4b-sdnq\model_index.json'
foreach ($required in @($detectPython, $fluxPython, $detectScript, $repaintScript, $modelIndex)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Text-removal workflow is missing: $required"
    }
}

$imagePath = (Get-Item -LiteralPath $Image).FullName
if ($imagePath.StartsWith('\\?\UNC\')) {
    $imagePath = '\\' + $imagePath.Substring(8)
} elseif ($imagePath.StartsWith('\\?\')) {
    $imagePath = $imagePath.Substring(4)
}
$outputPath = [IO.Path]::GetFullPath($Output)
if (Test-Path -LiteralPath $outputPath) {
    throw "Output already exists: $outputPath"
}

$env:PYTHONIOENCODING = 'utf-8'
$runDir = Join-Path $PSScriptRoot ('runs\text_' + [IO.Path]::GetFileNameWithoutExtension($imagePath) + '_' + [guid]::NewGuid().ToString('N').Substring(0, 8))
New-Item -ItemType Directory -Path $runDir -Force | Out-Null
$highlight = Join-Path $runDir 'detected_text.png'
$detections = Join-Path $runDir 'detected_text.json'

if ($IncludeVertical) {
    Write-Host 'Step 1/2: detecting horizontal and vertical card text...'
} else {
    Write-Host 'Step 1/2: detecting high-confidence horizontal card text...'
}
$detectArgs = @($detectScript, $imagePath, '--recognized-only', '--tight-boxes', '--output', $highlight)
if (-not $IncludeVertical) { $detectArgs += '--horizontal-only' }
& $detectPython @detectArgs
if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $detections -PathType Leaf)) {
    throw "Text detection failed. Intermediate files: $runDir"
}

Write-Host 'Step 2/2: repainting detected text with local FLUX.2 Klein...'
& $fluxPython $repaintScript $imagePath $detections --output $outputPath
if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $outputPath -PathType Leaf)) {
    throw "Text repaint failed. Intermediate files: $runDir"
}
Write-Host "Text-removal result: $outputPath"
