$ErrorActionPreference = 'Stop'

# This allowlist is the exact cleanup approved by the project owner.
$approvedFiles = @(
  'C:\Users\User\.cache\huggingface\hub\models--BAAI--bge-reranker-base\blobs\ced967c45fd1902eb92716c9ceeca7c95a936770ea9db611f5a841b926e33fbd.508469e8.incomplete',
  'C:\Users\User\.cache\huggingface\hub\models--BAAI--bge-reranker-base\blobs\ced967c45fd1902eb92716c9ceeca7c95a936770ea9db611f5a841b926e33fbd.98cf67f0.incomplete',
  'C:\Users\User\.cache\huggingface\hub\models--Qwen--Qwen2.5-VL-3B-Instruct\blobs\365531ff8752420e89dee707b79d021fb2d6e25abafe486f080555a4fe6972e4.incomplete',
  'C:\Users\User\.cache\huggingface\hub\models--Qwen--Qwen2.5-VL-3B-Instruct\blobs\41a8895c164b4d32bae6b302f4603fcbc1797f32dafa45c7e9bcda23c6755df8.incomplete',
  'C:\Users\User\.cache\huggingface\hub\models--Systran--faster-whisper-large-v3\blobs\69f74147e3334731bc3a76048724833325d2ec74642fb52620eda87352e3d4f1.incomplete',
  'C:\Users\User\.cache\huggingface\hub\models--Systran--faster-whisper-small\blobs\3e305921506d8872816023e4c273e75d2419fb89b24da97b4fe7bce14170d671.incomplete'
)
$approvedCache = 'C:\Users\User\AppData\Local\pip\Cache\http-v2'
$manifest = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'cleanup-candidates.json') -Raw | ConvertFrom-Json
$approvedPaths = @($approvedFiles) + @($approvedCache)
if ($manifest.items.Count -ne 7) { throw 'Cleanup manifest count changed.' }
foreach ($candidate in $manifest.items) {
  if ($approvedPaths -notcontains $candidate.path) { throw 'Cleanup manifest contains an unapproved path.' }
}

function Assert-NoReparseAncestors([string]$path) {
  $current = Get-Item -LiteralPath $path -Force
  while ($null -ne $current) {
    if (($current.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
      throw "Refusing reparse point: $($current.FullName)"
    }
    if ($current -is [IO.FileInfo]) { $current = $current.Directory }
    else { $current = $current.Parent }
  }
}

# Validate every existing target before removing anything.
foreach ($path in $approvedPaths) {
  if (-not (Test-Path -LiteralPath $path)) { continue }
  $resolved = (Resolve-Path -LiteralPath $path).ProviderPath
  if (-not [string]::Equals($resolved, $path, [StringComparison]::OrdinalIgnoreCase)) {
    throw "Resolved target differs from approved path: $path"
  }
  Assert-NoReparseAncestors $resolved
  $item = Get-Item -LiteralPath $resolved -Force
  if ($path -eq $approvedCache) {
    if (-not $item.PSIsContainer) { throw 'Expected cache directory.' }
    $cacheParent = [IO.Path]::GetFullPath('C:\Users\User\AppData\Local\pip\Cache') + '\'
    if (-not $resolved.StartsWith($cacheParent, [StringComparison]::OrdinalIgnoreCase)) {
      throw 'Cache is outside the approved cache root.'
    }
    $pending = [Collections.Generic.Stack[string]]::new()
    $pending.Push($resolved)
    while ($pending.Count -gt 0) {
      foreach ($child in Get-ChildItem -LiteralPath $pending.Pop() -Force) {
        if (($child.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
          throw "Refusing cache reparse point: $($child.FullName)"
        }
        if ($child.PSIsContainer) { $pending.Push($child.FullName) }
      }
    }
  } else {
    if ($item.PSIsContainer -or $item.Extension -ne '.incomplete') { throw 'Unexpected target type.' }
    $expected = @($manifest.items | Where-Object { $_.path -eq $path })
    if ($expected.Count -ne 1 -or $item.Length -ne $expected[0].bytes) {
      throw "Download size changed; not deleting: $path"
    }
    $recordedTime = [datetime]::Parse($expected[0].modified, [Globalization.CultureInfo]::InvariantCulture)
    if ([math]::Abs(($item.LastWriteTime - $recordedTime).TotalSeconds) -gt 0.01) {
      throw "Download modified since inspection; not deleting: $path"
    }
    if ($item.LastWriteTime -gt (Get-Date).AddDays(-1)) { throw "Recent download: $path" }
    $probe = [IO.File]::Open($resolved, [IO.FileMode]::Open, [IO.FileAccess]::Read, [IO.FileShare]::None)
    $probe.Dispose()
  }
}

$before = [IO.DriveInfo]::new('C:\').AvailableFreeSpace
$results = @()
foreach ($path in $approvedPaths) {
  if (-not (Test-Path -LiteralPath $path)) {
    $results += [pscustomobject]@{ path=$path; status='already_absent' }
    continue
  }
  try {
    if ($path -eq $approvedCache) { Remove-Item -LiteralPath $path -Recurse -Force }
    else { Remove-Item -LiteralPath $path -Force }
    if (Test-Path -LiteralPath $path) { throw 'Target still exists after removal.' }
    $results += [pscustomobject]@{ path=$path; status='deleted' }
  } catch {
    $results += [pscustomobject]@{ path=$path; status='error'; detail=$_.Exception.Message }
  }
}
$after = [IO.DriveInfo]::new('C:\').AvailableFreeSpace
$report = [pscustomobject]@{
  completed_at=(Get-Date).ToString('o')
  free_bytes_before=$before
  free_bytes_after=$after
  observed_free_bytes_increase=($after-$before)
  items=$results
}
$report | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $PSScriptRoot 'cleanup-result.json') -Encoding UTF8
$report | ConvertTo-Json -Depth 5
