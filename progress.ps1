# One-line answer to "how much is done?"
#
# Reads the vector table directly rather than the log, because the log is
# truncated every time the watchdog restarts the job - the table is the only
# source of truth that survives a restart.

$root = "c:\Users\User\Downloads\pubmed-ai"
$env:PUBMED_INDEX = "$root\index"

if (Test-Path "$root\index\vector_model.json") {
  Write-Host "EMBEDDING COMPLETE - restart the app to enable vector search:" -ForegroundColor Green
  Write-Host "  powershell -ExecutionPolicy Bypass -File `"$root\start_all.ps1`""
} else {
  python -c "import os,lancedb; t=lancedb.connect(os.environ['PUBMED_INDEX']).open_table('chunks_bge_small_en_v1_5'); n=t.count_rows(); left=238224-n; print('embedded {:,} / 238,224  ({:.1f}%)'.format(n, n/238224*100)); print('remaining {:,}  ~{:.1f} hours at 4.3/s'.format(left, left/4.3/3600))"
}

foreach ($j in @(@{n="embedding";p="*retrieval.py*vectors*";e="python.exe"},
                 @{n="watchdog ";p="*watchdog.ps1*";e="powershell.exe"},
                 @{n="web app  ";p="*app.py*";e="python.exe"})) {
  $r = Get-CimInstance Win32_Process -Filter "Name='$($j.e)'" -ErrorAction SilentlyContinue |
       Where-Object { $_.CommandLine -like $j.p }
  Write-Host "$($j.n) : $(if($r){'RUNNING'}else{'STOPPED'})"
}

if (Test-Path "$root\logs\watchdog.log") {
  Write-Host "`n--- watchdog log ---"
  Get-Content "$root\logs\watchdog.log" -Tail 5
}
