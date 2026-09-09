# Keeps the embedding pass alive overnight.
#
# The pass is resumable, so the safe response to any crash is simply to start it
# again - it skips whatever is already in the table. This watchdog does that
# without needing anyone awake. It stops on its own when the run completes,
# and gives up after MaxRestarts so a genuinely broken build cannot spin here
# forever.

$root  = "c:\Users\User\Downloads\pubmed-ai"
$log   = "$root\logs\watchdog.log"
$done  = "$root\index\vector_model.json"     # written only when the pass finishes
$MaxRestarts = 40
$restarts = 0

$env:PUBMED_STORE = "$root\store"
$env:PUBMED_INDEX = "$root\index"

function Say($msg) {
  $line = "{0}  {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $msg
  Add-Content -Path $log -Value $line -Encoding utf8
}

function EmbedRunning {
  $p = Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
       Where-Object { $_.CommandLine -like "*retrieval.py*vectors*" }
  return [bool]$p
}

Say "watchdog started (pid $PID)"

while ($true) {
  if (Test-Path $done) {
    Say "vector_model.json present - embedding COMPLETE. watchdog exiting."
    break
  }

  if (-not (EmbedRunning)) {
    if ($restarts -ge $MaxRestarts) {
      Say "hit MaxRestarts=$MaxRestarts without completing - giving up. Check embed.err."
      break
    }
    $restarts++
    Say "embedding process not found - restart #$restarts"
    Start-Process -FilePath "python" `
      -ArgumentList "retrieval.py","vectors","--model","BAAI/bge-small-en-v1.5","--backend","torch" `
      -WorkingDirectory "$root\pipeline" -WindowStyle Hidden `
      -RedirectStandardOutput "$root\logs\embed.log" `
      -RedirectStandardError  "$root\logs\embed.err"
    Start-Sleep -Seconds 60      # let it load the model before judging it again
  }

  Start-Sleep -Seconds 120
}
