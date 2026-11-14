<#
Experimental workflow: repair one original-size image, then upscale that result.
Requires the locally installed Upscayl standard-4x model and a supported GPU.
Only -Image is required. If -UpscaledOriginal is omitted, this script also
upscales the untouched input for outside-mask compositing.

Examples:
  .\Remove-Then-Upscale.ps1 -Image 'C:\Users\timun\Downloads\OP18-085.webp'
  .\Remove-Then-Upscale.ps1 -Image 'C:\Users\timun\Downloads\OP18-085.webp' `
    -UpscaledOriginal 'C:\Users\timun\Downloads\OP18-085_upscayl_2x_upscayl-standard-4x.png'
  .\Remove-Then-Upscale.ps1 -Image 'C:\Users\timun\Downloads\OP18-085.webp' `
    -UpscaleModel high-fidelity-4x
  .\Remove-Then-Upscale.ps1 -Image 'C:\Users\timun\Downloads\OP18-085.webp' -EdgeRecovery
  .\Remove-Then-Upscale.ps1 -Image 'C:\Users\timun\Downloads\batch_OP13-012_SP.png' -PaletteRecovery
#>
param(
    [Parameter(Mandatory = $true)]
    [string]$Image,
    [string]$UpscaledOriginal,
    [string]$Output,
    [string]$RunDirectory,
    [string]$Word = 'SAMPLE',
    [string]$Box,
    [ValidateRange(0.01, 0.95)]
    [double]$Opacity = 0.70,
    [ValidateSet('upscayl-standard-4x', 'high-fidelity-4x', 'digital-art-4x', 'remacri-4x', 'ultrasharp-4x')]
    [string]$UpscaleModel = 'upscayl-standard-4x',
    [string]$Font,
    [switch]$EdgeRecovery,
    [switch]$PaletteRecovery
)

$ErrorActionPreference = 'Stop'
$python = Join-Path $PSScriptRoot 'gpu_venv\Scripts\python.exe'
$wrapper = Join-Path $PSScriptRoot 'Remove-Watermark.ps1'
$compositor = Join-Path $PSScriptRoot 'composite_upscaled_repair.py'
$paletteScript = Join-Path $PSScriptRoot 'palette_inverse_probe.py'
$upscayl = 'C:\Program Files\Upscayl\resources\bin\upscayl-bin.exe'
$models = 'C:\Program Files\Upscayl\resources\models'
foreach ($required in @($python, $wrapper, $compositor, $upscayl)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Required tool not found: $required"
    }
}
if (-not (Test-Path -LiteralPath $models -PathType Container)) {
    throw "Upscayl models not found: $models"
}
if (-not (Test-Path -LiteralPath $Image -PathType Leaf)) {
    throw "Input image not found: $Image"
}
$imagePath = (Resolve-Path -LiteralPath $Image).ProviderPath
if ($imagePath.StartsWith('\\?\UNC\')) {
    $imagePath = '\\' + $imagePath.Substring(8)
} elseif ($imagePath.StartsWith('\\?\')) {
    $imagePath = $imagePath.Substring(4)
}
if ($PaletteRecovery) {
    if (-not (Test-Path -LiteralPath $paletteScript -PathType Leaf)) {
        throw "Palette recovery script not found: $paletteScript"
    }
    $sourceMode = & $python -c 'from PIL import Image; import sys; print(Image.open(sys.argv[1]).mode)' $imagePath
    if ($LASTEXITCODE -ne 0 -or $sourceMode -ne 'P') {
        throw '-PaletteRecovery currently requires an indexed (P-mode) original image.'
    }
}
if ($UpscaledOriginal) {
    if (-not (Test-Path -LiteralPath $UpscaledOriginal -PathType Leaf)) {
        throw "Upscaled original not found: $UpscaledOriginal"
    }
    $upscaledPath = (Resolve-Path -LiteralPath $UpscaledOriginal).ProviderPath
    if ($upscaledPath.StartsWith('\\?\UNC\')) {
        $upscaledPath = '\\' + $upscaledPath.Substring(8)
    } elseif ($upscaledPath.StartsWith('\\?\')) {
        $upscaledPath = $upscaledPath.Substring(4)
    }
}

$stem = [IO.Path]::GetFileNameWithoutExtension($imagePath)
if (-not $Output) {
    $outputPath = Join-Path $PSScriptRoot "${stem}_remove_then_upscale.png"
    $number = 2
    while (Test-Path -LiteralPath $outputPath) {
        $outputPath = Join-Path $PSScriptRoot "${stem}_remove_then_upscale_${number}.png"
        $number++
    }
} else {
    $outputPath = [IO.Path]::GetFullPath($Output)
    if (Test-Path -LiteralPath $outputPath) {
        throw "Output already exists: $outputPath"
    }
}

if ($RunDirectory) {
    $runDir = [IO.Path]::GetFullPath($RunDirectory)
} else {
    $runDir = Join-Path $PSScriptRoot ('runs\' + $stem + '_' + [guid]::NewGuid().ToString('N').Substring(0, 8))
}
if (Test-Path -LiteralPath $runDir) {
    throw "Intermediate directory already exists: $runDir"
}
New-Item -ItemType Directory -Path $runDir | Out-Null
$smallResult = Join-Path $runDir 'repaired_1x.png'
$smallMask = Join-Path $runDir 'repaired_1x_mask.png'
$candidate = Join-Path $runDir 'repaired_then_upscaled.png'

$wrapperArgs = @{
    Image = $imagePath
    Output = $smallResult
    Word = $Word
    Opacity = $Opacity
}
if ($Box) { $wrapperArgs.Box = $Box }
else { $wrapperArgs.AutoAcceptBox = $true }
if ($Font) { $wrapperArgs.Font = $Font }
if ($EdgeRecovery -or $PaletteRecovery) { $wrapperArgs.EdgeRecovery = $true }
if ($PaletteRecovery) {
    $wrapperArgs.SaveMatte = $true
    $wrapperArgs.CoreSize = 3
}
& $wrapper @wrapperArgs
if (-not (Test-Path -LiteralPath $smallResult -PathType Leaf)) {
    throw 'Original-size repair did not produce an output.'
}
if ($PaletteRecovery) {
    $paletteResult = Join-Path $runDir 'repaired_palette_1x.png'
    $mattePath = Join-Path $runDir 'repaired_1x_matte.png'
    Write-Host 'Recovering indexed colors within watermark strokes...'
    & $python $paletteScript --image $imagePath --reconstruction $smallResult `
        --matte $mattePath --mask $smallMask --output $paletteResult `
        --opacity $Opacity.ToString('0.00', [Globalization.CultureInfo]::InvariantCulture) `
        --strength 1.0 --spatial-weight 0.4 --all-interior
    if ($LASTEXITCODE -ne 0) { throw 'Palette recovery failed.' }
    $smallResult = $paletteResult
}

Write-Host 'Upscaling the repaired original-size image...'
& $upscayl -i $smallResult -o $candidate -m $models -n $UpscaleModel -s 2 -f png
if ($LASTEXITCODE -ne 0) { throw 'Upscayl failed on the repaired image.' }

if (-not $UpscaledOriginal) {
    $upscaledPath = Join-Path $runDir 'original_upscaled.png'
    Write-Host 'Upscaling the untouched original for outside-mask pixels...'
    & $upscayl -i $imagePath -o $upscaledPath -m $models -n $UpscaleModel -s 2 -f png
    if ($LASTEXITCODE -ne 0) { throw 'Upscayl failed on the original image.' }
}

& $python $compositor --original-upscaled $upscaledPath --candidate $candidate `
    --small-mask $smallMask --output $outputPath
if ($LASTEXITCODE -ne 0) { throw 'Compositing failed.' }
Write-Host "Final result: $outputPath"
Write-Host "Intermediate files: $runDir"
