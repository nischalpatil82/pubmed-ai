param(
    [string]$Source = 'C:\Users\User\Downloads\pubmed25n1509 (1).zip',
    [string]$OutputRoot = "$PSScriptRoot\covid-files",
    [int]$MaxFiles = 0,
    [switch]$Sample
)
$ErrorActionPreference = 'Stop'
$buildArgs = @('--src', $Source, '--root', $OutputRoot)
if (Test-Path -LiteralPath "$PSScriptRoot\raw\desc2026.xml") { $buildArgs += @('--mesh', "$PSScriptRoot\raw\desc2026.xml") }
if ($MaxFiles -gt 0) { $buildArgs += @('--max-files', $MaxFiles) }
if ($Sample) { $buildArgs += '--sample' }
python "$PSScriptRoot\pipeline\stream_store.py" @buildArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python "$PSScriptRoot\pipeline\manage.py" bm25 --dataset "$OutputRoot\dataset.json"
exit $LASTEXITCODE
