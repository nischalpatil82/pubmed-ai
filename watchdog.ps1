param([string]$Dataset = "$PSScriptRoot\covid-files\dataset.json", [int]$MaxRestarts = 3)
$ErrorActionPreference = 'Stop'
# OS writer locks prevent duplicate writers. A successful limited build is not complete.
for ($attempt = 0; $attempt -lt $MaxRestarts; $attempt++) {
    python "$PSScriptRoot\pipeline\manage.py" vectors --dataset "$Dataset"
    if ($LASTEXITCODE -eq 0) { exit 0 }
    if ($attempt + 1 -lt $MaxRestarts) { Start-Sleep -Seconds 30 }
}
throw 'Embedding stopped after the retry limit; inspect the error before restarting.'
