param([string]$Dataset = '', [int]$Port = 8010)
$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$TaskName = 'PubMed AI local sync'
$PythonPath = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
$Supervisor = Join-Path $PSScriptRoot 'local_sync.py'
if (!$Dataset) { $Dataset = Join-Path $ProjectRoot 'covid-files\dataset.json' }
$Dataset = [System.IO.Path]::GetFullPath($Dataset)
if (!(Test-Path -LiteralPath $PythonPath -PathType Leaf)) { throw 'Install the project virtual environment first.' }
if (!(Test-Path -LiteralPath $Dataset -PathType Leaf)) { throw 'The local dataset manifest is missing.' }
if ($Port -lt 1 -or $Port -gt 65535) { throw 'Port must be between 1 and 65535.' }
$Existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($Existing) {
    if ($Existing.Actions.Execute -ne $PythonPath -or $Existing.Actions.Arguments -notlike '*local_sync.py*') {
        throw 'A different task uses this name; review it before replacing it.'
    }
    if ($Existing.State -eq 'Running') {
        throw 'Stop the existing PubMed AI local sync task, then run this command again to restart with the updated supervisor.'
    }
}
$Socket = [System.Net.Sockets.TcpClient]::new()
try {
    $Attempt = $Socket.ConnectAsync('127.0.0.1', $Port)
    try { [void]$Attempt.Wait(1000) } catch { }
    if ($Socket.Connected) { throw "Port $Port is occupied. Stop the manual app before enabling automatic updates." }
} finally { $Socket.Dispose() }
$WindowsAccount = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
$Action = New-ScheduledTaskAction -Execute $PythonPath -Argument "`"$Supervisor`" --dataset `"$Dataset`" --port $Port" -WorkingDirectory $ProjectRoot
$Trigger = New-ScheduledTaskTrigger -AtLogOn -User $WindowsAccount
$Principal = New-ScheduledTaskPrincipal -UserId $WindowsAccount -LogonType Interactive -RunLevel Limited
$Settings = New-ScheduledTaskSettingsSet -MultipleInstances IgnoreNew -ExecutionTimeLimit ([TimeSpan]::Zero) -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1)
Register-ScheduledTask -TaskName $TaskName -Action $Action -Trigger $Trigger -Principal $Principal -Settings $Settings -Description 'Serve PubMed Intelligence and apply tested GitHub code updates every five minutes.' -Force | Out-Null
Start-ScheduledTask -TaskName $TaskName
Write-Host 'Automatic updates enabled for this Windows account. Check ops\logs\sync.log for readiness and any skipped updates.'
Write-Host 'The PC must remain on, online, and signed in. Initial performance preparation may take several minutes.'
