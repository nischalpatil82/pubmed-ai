# Add new XML files to the system.
#
#   1. drop new .xml.gz files into raw\
#   2. run:  powershell -ExecutionPolicy Bypass -File add_files.ps1
#
# Safe to re-run. Everything is rebuilt from ALL the .xml.gz files present, so
# duplicates are removed and nothing is double counted. Only the embedding step
# is incremental - it skips articles already embedded and does just the new
# ones, which is the difference between minutes and another 15 hours.

$root = "c:\Users\User\Downloads\pubmed-ai"
$env:PUBMED_STORE = "$root\store"
$env:PUBMED_INDEX = "$root\index"

$files = @(Get-ChildItem "$root\raw\*.xml.gz" -ErrorAction SilentlyContinue)
if ($files.Count -eq 0) { exit "no .xml.gz files found in $root" }
Write-Host "found $($files.Count) .xml.gz files" -ForegroundColor Cyan

# stop the app so it is not reading tables while they are rewritten
$app = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
       Where-Object { $_.CommandLine -like "*app.py*" }
if ($app) { Stop-Process -Id $app.ProcessId -Force; Write-Host "stopped web app" }

Push-Location "$root\pipeline"

Write-Host "`n[1/4] XML -> CSV" -ForegroundColor Yellow
python pubmed_ingest.py --src "$root\raw" --out "$root\csv" --workers 4
if ($LASTEXITCODE -ne 0) { Pop-Location; exit "ingest failed" }

Write-Host "`n[2/4] CSV -> Parquet (dedups across all files)" -ForegroundColor Yellow
python build_store.py --csv "$root\csv" --out "$root\store"
if ($LASTEXITCODE -ne 0) { Pop-Location; exit "build_store failed" }

Write-Host "`n[3/4] keyword index (full rebuild, ~2 min)" -ForegroundColor Yellow
python retrieval.py bm25
if ($LASTEXITCODE -ne 0) { Pop-Location; exit "bm25 failed" }

Write-Host "`n[4/4] embeddings (only the NEW articles)" -ForegroundColor Yellow
# vector_model.json means "finished". Remove it so the pass runs again for the
# new rows; build_vectors rewrites it when it completes.
if (Test-Path "$root\index\vector_model.json") { Remove-Item "$root\index\vector_model.json" }
python retrieval.py vectors --model BAAI/bge-small-en-v1.5 --backend torch

Pop-Location

Write-Host "`nrestarting web app..." -ForegroundColor Cyan
Start-Process -FilePath "python" -ArgumentList "app.py","--port","8010" `
  -WorkingDirectory "$root\pipeline" -WindowStyle Hidden `
  -RedirectStandardOutput "$root\logs\app.log" -RedirectStandardError "$root\logs\app.err"
Write-Host "done -> http://127.0.0.1:8010" -ForegroundColor Green
