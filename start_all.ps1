param([string]$Dataset = "$PSScriptRoot\covid-files\dataset.json", [int]$Port = 8010)
$ErrorActionPreference = 'Stop'

# Optional literal assignments only: never dot-source/evaluate this file or log values.
# Existing process environment wins (case-insensitive, including empty entries).
$EnvFile = Join-Path $PSScriptRoot 'llm.env'
$ExistingNames = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::OrdinalIgnoreCase)
foreach ($Name in [Environment]::GetEnvironmentVariables('Process').Keys) {
    [void]$ExistingNames.Add([string]$Name)
}
if (Test-Path -LiteralPath $EnvFile -PathType Leaf) {
    $LineNumber = 0
    foreach ($Line in [System.IO.File]::ReadAllLines($EnvFile)) {
        $LineNumber++
        if ($Line -match '^\s*(#.*)?$') { continue }
        if ($Line -cnotmatch '^\s*(PUBMED_[A-Z0-9_]+)\s*=(.*)$') {
            Write-Warning "Ignored unsupported llm.env entry at line $LineNumber."
            continue
        }
        $Name = $Matches[1]
        $Value = $Matches[2].Trim()
        if ($ExistingNames.Contains($Name)) { continue }
        if ($Value.StartsWith('"') -or $Value.StartsWith("'")) {
            if ($Value.Length -lt 2 -or $Value[-1] -ne $Value[0]) {
                Write-Warning "Ignored malformed llm.env entry at line $LineNumber."
                continue
            }
            $Value = $Value.Substring(1, $Value.Length - 2)
        }
        # No interpolation, shell expansion, alias mapping, or key-based cloud opt-in.
        try {
            [Environment]::SetEnvironmentVariable($Name, $Value, 'Process')
            [void]$ExistingNames.Add($Name)
        } catch {
            throw "Could not load llm.env entry at line $LineNumber."
        }
    }
}
# Serving and embedding are explicit separate jobs. No global process termination.
python "$PSScriptRoot\pipeline\manage.py" serve --dataset "$Dataset" --port $Port
exit $LASTEXITCODE
