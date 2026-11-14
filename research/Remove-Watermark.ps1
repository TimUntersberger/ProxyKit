<#
Run the local watermark reconstruction with automatic SAMPLE box detection,
prompts, and safe output naming.

Examples:
  .\Remove-Watermark.ps1
  .\Remove-Watermark.ps1 -Image 'C:\Users\timun\Downloads\OP18-021.webp'
  .\Remove-Watermark.ps1 -Image 'C:\Users\timun\Downloads\OP18-021.webp' -AutoAcceptBox
  .\Remove-Watermark.ps1 -Image 'C:\Users\timun\Downloads\OP18-021.webp' -EdgeRecovery
  .\Remove-Watermark.ps1 -Image 'C:\path\card.png' -Box '54,379,588,521' -Opacity 0.70
#>
param(
    [string]$Image,
    [string]$Output,
    [string]$Word = 'SAMPLE',
    [string]$Box,
    [ValidateRange(0.01, 0.95)]
    [double]$Opacity = 0.70,
    [string]$Font,
    [ValidateRange(0, 100)]
    [double]$FeatureAnchorStrength = 8.0,
    [ValidateSet(3, 5, 7, 9)]
    [int]$CoreSize = 7,
    [switch]$AutoAcceptBox,
    [switch]$NoTextureRecovery,
    [switch]$EdgeRecovery,
    [switch]$SaveMatte
)

$ErrorActionPreference = 'Stop'
$pythonPath = Join-Path $PSScriptRoot 'gpu_venv\Scripts\python.exe'
$scriptPath = Join-Path $PSScriptRoot 'one_image_poisson.py'
$detectorPath = Join-Path $PSScriptRoot 'detect_watermark_box.py'
if (-not (Test-Path -LiteralPath $pythonPath -PathType Leaf)) {
    throw "Python environment not found at $pythonPath"
}
if (-not (Test-Path -LiteralPath $scriptPath -PathType Leaf)) {
    throw "Reconstruction script not found at $scriptPath"
}

if ([string]::IsNullOrWhiteSpace($Image)) {
    $Image = Read-Host 'Image path (you can drag the file into this window)'
}
$Image = $Image.Trim().Trim('"').Trim("'")
if (-not (Test-Path -LiteralPath $Image -PathType Leaf)) {
    throw "Image not found: $Image"
}
$imagePath = (Resolve-Path -LiteralPath $Image).ProviderPath
if ($imagePath.StartsWith('\\?\UNC\')) {
    $imagePath = '\\' + $imagePath.Substring(8)
} elseif ($imagePath.StartsWith('\\?\')) {
    $imagePath = $imagePath.Substring(4)
}

# This proportional box is shown only as a fallback if detection is uncertain.
$sizeText = & $pythonPath -c 'from PIL import Image; import sys; print(*Image.open(sys.argv[1]).size)' $imagePath
if ($LASTEXITCODE -ne 0 -or $sizeText -notmatch '^(\d+)\s+(\d+)$') {
    throw 'Could not read the image dimensions.'
}
$width = [int]$Matches[1]
$height = [int]$Matches[2]
$suggested = @(
    [int][Math]::Round(51 * $width / 600),
    [int][Math]::Round(355 * $height / 838),
    [int][Math]::Round(551 * $width / 600),
    [int][Math]::Round(488 * $height / 838)
) -join ','
Write-Host "Image: $width x $height pixels"
if ([string]::IsNullOrWhiteSpace($Box)) {
    if (-not (Test-Path -LiteralPath $detectorPath -PathType Leaf)) {
        throw "Watermark detector not found at $detectorPath"
    }
    $detectorArgs = @($detectorPath, '--image', $imagePath, '--word', $Word)
    if ($Font) { $detectorArgs += @('--font', $Font) }
    $detectorOutput = @(& $pythonPath @detectorArgs)
    if ($LASTEXITCODE -ne 0 -or $detectorOutput.Count -lt 2 -or
        $detectorOutput[1] -notmatch '^score\s+([0-9.]+)$') {
        throw 'Automatic watermark detection failed. Supply -Box manually.'
    }
    $detectedScore = [double]::Parse($Matches[1], [Globalization.CultureInfo]::InvariantCulture)
    $detectedBox = $detectorOutput[0]
    if (-not $Font -and $detectorOutput.Count -ge 3 -and $detectorOutput[2] -match '^font\s+(.+)$') {
        $Font = $Matches[1]
    }
    if ($detectedScore -ge 0.5) {
        Write-Host "Detected $Word at box $detectedBox using $Font (match score $detectedScore)."
        if ($AutoAcceptBox) {
            $Box = $detectedBox
        } else {
            $entered = Read-Host "Watermark box left,top,right,bottom [$detectedBox]"
            $Box = if ([string]::IsNullOrWhiteSpace($entered)) { $detectedBox } else { $entered.Trim() }
        }
    } else {
        Write-Warning "No confident $Word match found (score $detectedScore)."
        if ($AutoAcceptBox) {
            throw "Automatic watermark detection is too uncertain (score $detectedScore). Supply -Box and -Font manually."
        } else {
            $entered = Read-Host "Enter a watermark box left,top,right,bottom (layout suggestion: $suggested)"
            if ([string]::IsNullOrWhiteSpace($entered)) {
                throw 'A box is required when detection confidence is low.'
            }
            $Box = $entered.Trim()
        }
    }
}
$parts = $Box -split '\s*,\s*'
if ($parts.Count -ne 4) {
    throw 'Box must contain four comma-separated integers: left,top,right,bottom.'
}
try {
    $left = [int]$parts[0]
    $top = [int]$parts[1]
    $right = [int]$parts[2]
    $bottom = [int]$parts[3]
} catch {
    throw 'Box must contain four comma-separated integers.'
}
if ($left -lt 0 -or $top -lt 0 -or $right -gt $width -or $bottom -gt $height -or
    $left -ge $right -or $top -ge $bottom) {
    throw "Box must be inside the ${width} x ${height} image, with left < right and top < bottom."
}
$Box = "$left,$top,$right,$bottom"

if ([string]::IsNullOrWhiteSpace($Output)) {
    $stem = [IO.Path]::GetFileNameWithoutExtension($imagePath)
    $outputPath = Join-Path $PSScriptRoot "${stem}_repaired.png"
    $number = 2
    while (Test-Path -LiteralPath $outputPath) {
        $outputPath = Join-Path $PSScriptRoot "${stem}_repaired_${number}.png"
        $number++
    }
} else {
    $outputPath = [IO.Path]::GetFullPath($Output)
    if (Test-Path -LiteralPath $outputPath) {
        throw "Output already exists: $outputPath"
    }
}

$pythonArgs = @(
    $scriptPath, '--image', $imagePath, '--output', $outputPath,
    '--word', $Word, '--box', $Box, '--opacity',
    $Opacity.ToString('0.00', [Globalization.CultureInfo]::InvariantCulture),
    '--feature-anchor-strength',
    $FeatureAnchorStrength.ToString('0.###', [Globalization.CultureInfo]::InvariantCulture),
    '--core-size', $CoreSize.ToString()
)
if ($Font) { $pythonArgs += @('--font', $Font) }
if (-not $NoTextureRecovery) { $pythonArgs += '--texture-recovery' }
if ($EdgeRecovery) { $pythonArgs += @('--edge-gradient-weight', '0.35', '--edge-clearance', '0.5') }
if ($SaveMatte) { $pythonArgs += '--save-matte' }

Write-Host "Saving to: $outputPath"
& $pythonPath @pythonArgs
if ($LASTEXITCODE -ne 0) {
    throw "Reconstruction failed with exit code $LASTEXITCODE."
}
$detectionPath = [IO.Path]::ChangeExtension($outputPath, '.detection.json')
@{ box = $Box; font = $Font; score = $detectedScore } |
    ConvertTo-Json -Compress | Set-Content -LiteralPath $detectionPath -Encoding UTF8
Write-Host "Done. Result: $outputPath"
Write-Host 'A repair-mask PNG and detected-anchor PNG were saved beside it.'
