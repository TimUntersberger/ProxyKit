<#
Remove a SAMPLE watermark with source repair plus local FLUX.2 Klein 4B.
The final image is 2x the input size and defaults to the input folder.
Intermediate images remain in runs/.

Examples:
  .\Remove-With-Flux2.ps1 -Image 'C:\Users\timun\Downloads\OP18-085.webp'
  .\Remove-With-Flux2.ps1 -Image 'C:\Users\timun\Downloads\OP18-021.webp' -Output '.\franky_clean.png'

Current local Klein crop supports images 512+ pixels high and at most 640 wide.
Python is used only for this feasibility workflow, not the planned desktop app.
#>
param(
    [Parameter(Position = 0)]
    [string]$Image,
    [string]$Output,
    [string]$Box,
    [ValidateRange(0.01, 0.95)]
    [double]$Opacity = 0.70,
    [ValidateRange(0, 2147483647)]
    [int]$Seed = 101,
    [string]$UpscaledOriginal,
    [ValidateSet('auto', 'repaired', 'original')]
    [string]$ModelReference = 'auto',
    [ValidateSet('repaired', 'edge-inpaint', 'paired-edge-inpaint')]
    [string]$ConditioningMode = 'repaired'
)

$ErrorActionPreference = 'Stop'
$sourceRepair = Join-Path $PSScriptRoot 'Remove-Then-Upscale.ps1'
$fluxScript = Join-Path $PSScriptRoot 'flux2_repaint_card.py'
$sourcePython = Join-Path $PSScriptRoot 'gpu_venv\Scripts\python.exe'
$fluxPython = Join-Path $PSScriptRoot 'flux_compat_venv\Scripts\python.exe'
$modelIndex = Join-Path $PSScriptRoot 'models\flux2-klein-4b-sdnq\model_index.json'
foreach ($required in @($sourceRepair, $fluxScript, $sourcePython, $fluxPython, $modelIndex)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Required workflow file is missing: $required"
    }
}

if ([string]::IsNullOrWhiteSpace($Image)) {
    $Image = Read-Host 'Image path (you can drag the file here)'
}
$Image = $Image.Trim().Trim('"').Trim("'")
if (-not (Test-Path -LiteralPath $Image -PathType Leaf)) {
    throw "Input image not found: $Image"
}
$imagePath = (Resolve-Path -LiteralPath $Image).ProviderPath
if ($imagePath.StartsWith('\\?\UNC\')) {
    $imagePath = '\\' + $imagePath.Substring(8)
} elseif ($imagePath.StartsWith('\\?\')) {
    $imagePath = $imagePath.Substring(4)
}
$sizeText = & $sourcePython -c 'from PIL import Image; import sys; print(*Image.open(sys.argv[1]).size)' $imagePath
if ($LASTEXITCODE -ne 0 -or $sizeText -notmatch '^(\d+)\s+(\d+)$') {
    throw 'Could not read the image dimensions.'
}
$width = [int]$Matches[1]
$height = [int]$Matches[2]
if ($width -gt 640 -or $height -lt 512) {
    throw "The current Klein workflow needs an image at most 640 pixels wide and at least 512 pixels high; got ${width}x${height}."
}

$stem = [IO.Path]::GetFileNameWithoutExtension($imagePath)
if ($Output) {
    $outputPath = [IO.Path]::GetFullPath($Output)
    if (Test-Path -LiteralPath $outputPath) {
        throw "Output already exists: $outputPath"
    }
} else {
    $outputPath = Join-Path (Split-Path -Parent $imagePath) "${stem}_flux2_klein_2x.png"
    $number = 2
    while (Test-Path -LiteralPath $outputPath) {
        $outputPath = Join-Path (Split-Path -Parent $imagePath) "${stem}_flux2_klein_2x_${number}.png"
        $number++
    }
}

$runDir = Join-Path $PSScriptRoot ('runs\' + $stem + '_flux2_' + [guid]::NewGuid().ToString('N').Substring(0, 8))
$baselinePath = Join-Path $runDir 'source_baseline_2x.png'
$repairArgs = @{
    Image = $imagePath
    Output = $baselinePath
    RunDirectory = $runDir
    Opacity = $Opacity
}
if ($Box) { $repairArgs.Box = $Box }
if ($UpscaledOriginal) { $repairArgs.UpscaledOriginal = $UpscaledOriginal }

Write-Host 'Step 1/2: reconstructing and upscaling the source image...'
& $sourceRepair @repairArgs
if (-not (Test-Path -LiteralPath $baselinePath -PathType Leaf)) {
    throw 'Source repair did not produce the 2x baseline.'
}

$repairedPath = Join-Path $runDir 'repaired_1x.png'
$maskPath = Join-Path $runDir 'repaired_1x_mask.png'
$detectionPath = Join-Path $runDir 'repaired_1x.detection.json'
$modelInput = $repairedPath
$modelBaseline = $baselinePath
if ($ModelReference -eq 'original' -or ($ModelReference -eq 'auto' -and
    (Test-Path -LiteralPath $detectionPath) -and
    ((Get-Content -LiteralPath $detectionPath -Raw | ConvertFrom-Json).font -match '(?i)\\arial\.ttf$'))) {
    $modelInput = $imagePath
    $modelBaseline = if ($UpscaledOriginal) { $UpscaledOriginal } else { Join-Path $runDir 'original_upscaled.png' }
    if (-not (Test-Path -LiteralPath $modelBaseline -PathType Leaf)) {
        throw 'Original upscale is required for direct model reference.'
    }
    Write-Host 'Using the original as the model reference for thin watermark lettering.'
}
$prefix = Join-Path $runDir 'flux2'
$fluxArgs = @(
    $fluxScript,
    '--repaired', $modelInput,
    '--mask', $maskPath,
    '--baseline-2x', $modelBaseline,
    '--output-prefix', $prefix,
    '--seed', $Seed.ToString([Globalization.CultureInfo]::InvariantCulture),
    '--conditioning-mode', $ConditioningMode
)
Write-Host 'Step 2/2: repainting the watermark with local FLUX.2 Klein...'
& $fluxPython @fluxArgs
if ($LASTEXITCODE -ne 0) {
    throw "FLUX.2 repaint failed; intermediate files are in $runDir"
}
$finalPath = Join-Path $runDir 'flux2_full_2x.png'
if (-not (Test-Path -LiteralPath $finalPath -PathType Leaf)) {
    throw "FLUX.2 did not produce the final image: $finalPath"
}
$outputParent = Split-Path -Parent $outputPath
if ($outputParent -and -not (Test-Path -LiteralPath $outputParent -PathType Container)) {
    New-Item -ItemType Directory -Path $outputParent -Force | Out-Null
}
Copy-Item -LiteralPath $finalPath -Destination $outputPath
Write-Host "Final image: $outputPath"
Write-Host "Intermediates: $runDir"
