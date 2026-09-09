# Restore everything after a reboot, a flat battery, or a bad wake from sleep.
#
# Safe to run any time: it only starts what is not already running, so running
# it twice does nothing the second time. Nothing here loses work - the
# embedding pass resumes from whatever is already on disk.
#
#   powershell -ExecutionPolicy Bypass -File start_all.ps1

$root = "c:\Users\User\Downloads\pubmed-ai"
$env:PUBMED_STORE = "$root\store"
$env:PUBMED_INDEX = "$root\index"

# ---------------------------------------------------------------- LLM choice
# Put your key in llm.env (one line: PUBMED_API_KEY=gsk_...) and it is picked up
# here. Keeping it in a separate file means the key never lands in a script you
# might paste into chat, a screenshot, or a git commit.
if (Test-Path "$root\llm.env") {
  Get-Content "$root\llm.env" | ForEach-Object {
    if ($_ -match '^\s*([A-Z_]+)\s*=\s*(.+?)\s*$') {
      Set-Item -Path "env:$($Matches[1])" -Value $Matches[2]
    }
  }
  if ($env:PUBMED_API_KEY) {
    if (-not $env:PUBMED_LLM) { $env:PUBMED_LLM = "cloud" }
    Write-Host "LLM       : cloud ($($env:PUBMED_CLOUD_MODEL))" -ForegroundColor Green
  }
} else {
  Write-Host "LLM       : local ollama (no llm.env found)" -ForegroundColor DarkGray
}

function Running($pattern, $exe) {
  $p = Get-CimInstance Win32_Process -Filter "Name='$exe'" -ErrorAction SilentlyContinue |
       Where-Object { $_.CommandLine -like $pattern }
  return [bool]$p
}

# ---------------------------------------------------------------- embedding
if (Test-Path "$root\index\vector_model.json") {
  Write-Host "embedding : COMPLETE (vector_model.json present)" -ForegroundColor Green
}
elseif (Running "*retrieval.py*vectors*" "python.exe") {
  Write-Host "embedding : already running" -ForegroundColor Green
}
else {
  Start-Process -FilePath "python" `
    -ArgumentList "retrieval.py","vectors","--model","BAAI/bge-small-en-v1.5","--backend","torch" `
    -WorkingDirectory "$root\pipeline" -WindowStyle Hidden `
    -RedirectStandardOutput "$root\logs\embed.log" -RedirectStandardError "$root\logs\embed.err"
  Write-Host "embedding : STARTED (resumes from what is already embedded)" -ForegroundColor Yellow
}

# ---------------------------------------------------------------- watchdog
if (Running "*watchdog.ps1*" "powershell.exe") {
  Write-Host "watchdog  : already running" -ForegroundColor Green
}
elseif (Test-Path "$root\index\vector_model.json") {
  Write-Host "watchdog  : not needed, embedding is done" -ForegroundColor Green
}
else {
  Start-Process -FilePath "powershell.exe" `
    -ArgumentList "-NoProfile","-ExecutionPolicy","Bypass","-File","$root\watchdog.ps1" `
    -WindowStyle Hidden
  Write-Host "watchdog  : STARTED" -ForegroundColor Yellow
}

# ---------------------------------------------------------------- web app
if (Running "*app.py*" "python.exe") {
  Write-Host "web app   : already running -> http://127.0.0.1:8010" -ForegroundColor Green
}
else {
  Start-Process -FilePath "python" -ArgumentList "app.py","--port","8010" `
    -WorkingDirectory "$root\pipeline" -WindowStyle Hidden `
    -RedirectStandardOutput "$root\logs\app.log" -RedirectStandardError "$root\logs\app.err"
  Write-Host "web app   : STARTED -> http://127.0.0.1:8010" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Give the app ~25 seconds, then open http://127.0.0.1:8010"
Write-Host "Progress:  powershell -File `"$root\progress.ps1`""
