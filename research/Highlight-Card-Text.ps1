param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$ImagePath,
    [string]$OutputPath,
    [switch]$HorizontalOnly,
    [switch]$RecognizedOnly,
    [switch]$TightBoxes,
    [ValidateRange(0.01, 0.99)]
    [double]$Threshold = 0.4
)

$ErrorActionPreference = 'Stop'
$python = Join-Path $PSScriptRoot 'text_detect_venv\Scripts\python.exe'
$script = Join-Path $PSScriptRoot 'highlight_card_text.py'
if (-not (Test-Path -LiteralPath $python)) {
    throw "Text detector environment is missing: $python"
}
$inputFile = (Get-Item -LiteralPath $ImagePath).FullName
if ($inputFile.StartsWith('\\?\UNC\')) {
    $inputFile = '\\' + $inputFile.Substring(8)
} elseif ($inputFile.StartsWith('\\?\')) {
    $inputFile = $inputFile.Substring(4)
}
$arguments = @($script, $inputFile, '--threshold', $Threshold.ToString([Globalization.CultureInfo]::InvariantCulture))
if ($OutputPath) {
    $arguments += @('--output', [System.IO.Path]::GetFullPath($OutputPath))
}
if ($HorizontalOnly) {
    $arguments += '--horizontal-only'
}
if ($RecognizedOnly) {
    $arguments += '--recognized-only'
}
if ($TightBoxes) {
    $arguments += '--tight-boxes'
}
& $python @arguments
if ($LASTEXITCODE -ne 0) {
    throw "Text detection failed with exit code $LASTEXITCODE"
}
