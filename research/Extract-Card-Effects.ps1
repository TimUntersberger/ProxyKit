param(
    [Parameter(Mandatory = $true)] [string]$Image,
    [Parameter(Mandatory = $true)] [string]$Output
)

$ErrorActionPreference = 'Stop'
$python = Join-Path $PSScriptRoot 'text_detect_venv\Scripts\python.exe'
$detector = Join-Path $PSScriptRoot 'highlight_card_text.py'
$extractor = Join-Path $PSScriptRoot 'extract_effect_text.py'
foreach ($required in @($python, $detector, $extractor)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Effect extraction is missing: $required"
    }
}
$imagePath = (Get-Item -LiteralPath $Image).FullName
if ($imagePath.StartsWith('\\?\UNC\')) {
    $imagePath = '\\' + $imagePath.Substring(8)
} elseif ($imagePath.StartsWith('\\?\')) {
    $imagePath = $imagePath.Substring(4)
}
$outputPath = [IO.Path]::GetFullPath($Output)
if (Test-Path -LiteralPath $outputPath) { throw "Output already exists: $outputPath" }
$runDir = Join-Path $PSScriptRoot ('runs\effects_' + [IO.Path]::GetFileNameWithoutExtension($imagePath) + '_' + [guid]::NewGuid().ToString('N').Substring(0, 8))
New-Item -ItemType Directory -Path $runDir | Out-Null
$preview = Join-Path $runDir 'detected_text.png'
$manifest = Join-Path $runDir 'detected_text.json'
$env:PYTHONIOENCODING = 'utf-8'
Write-Host 'Detecting card text...'
& $python $detector $imagePath --horizontal-only --recognized-only --output $preview
if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $manifest)) { throw 'Card text detection failed.' }
Write-Host 'Extracting effect lines...'
& $python $extractor $manifest --output $outputPath
if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $outputPath)) { throw 'Effect extraction failed.' }
Write-Host "Effect draft: $outputPath"
Write-Host "Detection preview: $preview"
