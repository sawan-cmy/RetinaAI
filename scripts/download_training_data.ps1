param(
    [string]$Competition = "aptos2019-blindness-detection",
    [string]$OutputDir = "data/raw/aptos2019"
)

$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

python -m kaggle competitions download -c $Competition -p $OutputDir

$zipPath = Join-Path $OutputDir "$Competition.zip"
if (Test-Path -LiteralPath $zipPath) {
    Expand-Archive -LiteralPath $zipPath -DestinationPath $OutputDir -Force
    Write-Host "Downloaded and extracted $Competition into $OutputDir"
} else {
    Write-Host "Download finished, but expected zip was not found at $zipPath"
}
