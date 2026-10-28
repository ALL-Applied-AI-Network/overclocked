param([switch]$BatteryMock)
$ErrorActionPreference = 'Stop'
$project = Split-Path -Parent $PSScriptRoot
$privateDir = Join-Path $project ('logs\validation-local-' + (Get-Date -Format 'yyyyMMdd-HHmmss'))
New-Item -ItemType Directory -Path $privateDir -Force | Out-Null
$uv = Get-Command uv -ErrorAction SilentlyContinue
if (-not $uv) {
    '{"status":"blocked","reason":"uv not installed/on PATH","python_checks_executed":false}' | Set-Content -LiteralPath (Join-Path $privateDir 'summary.json')
    throw 'uv is unavailable. Install it from the official Astral documentation, reopen PowerShell, and rerun this script.'
}
$names = @('TRIAL_MODES','TARGET_OBJECT','ADAPTIVE_OFFSET','GRASP_FRAME','SKIP_WRIST_XY','GRASP_OFFSETS','CUBE_DETECTOR','CAPTURE_ONLY','CAPTURE_DEPTH','PYTHONUTF8','PYTHONIOENCODING')
$saved = @{}
foreach ($name in $names) {
    $saved[$name] = [Environment]::GetEnvironmentVariable($name, 'Process')
    [Environment]::SetEnvironmentVariable($name, $null, 'Process')
}
$records = [System.Collections.Generic.List[object]]::new()
function Run-LocalCheck([string]$Name, [string[]]$Arguments) {
    $logPath = Join-Path $privateDir ($Name + '.txt')
    $stdoutPath = Join-Path $privateDir ($Name + '.stdout.txt')
    $stderrPath = Join-Path $privateDir ($Name + '.stderr.txt')
    Write-Output ('Running local check: ' + $Name)
    $process = Start-Process -FilePath $uv.Source -ArgumentList $Arguments -WorkingDirectory $project -WindowStyle Hidden -Wait -PassThru -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath
    $code = $process.ExitCode
    $stdout = Get-Content -Raw -Encoding UTF8 -LiteralPath $stdoutPath
    $stderr = Get-Content -Raw -Encoding UTF8 -LiteralPath $stderrPath
    $combined = "STDOUT:`n$stdout`nSTDERR:`n$stderr"
    $combined | Set-Content -Encoding UTF8 -LiteralPath $logPath
    if ($stdout) { Write-Output $stdout }
    if ($stderr) { Write-Output $stderr }
    $records.Add([pscustomobject]@{check=$Name; exit_code=$code; log_filename=($Name+'.txt')})
    $records | ConvertTo-Json -Depth 3 | Set-Content -LiteralPath (Join-Path $privateDir 'summary.json')
    if ($code -ne 0) { throw ('Local check failed: ' + $Name + '; see private log.') }
}
Push-Location -LiteralPath $project
try {
    $env:PYTHONUTF8='1'
    $env:PYTHONIOENCODING='utf-8'
    Run-LocalCheck 'uv-sync' @('sync')
    Run-LocalCheck 'vision' @('run','test_vision.py')
    Run-LocalCheck 'mock' @('run','run.py','--mock','--trials','1','--trial-s','40')
    if ($BatteryMock) {
        $env:TRIAL_MODES='B'; $env:TARGET_OBJECT='battery'; $env:ADAPTIVE_OFFSET='1'; $env:GRASP_FRAME='gripper'
        Run-LocalCheck 'battery-mock' @('run','run.py','--mock','--trials','1','--trial-s','40')
    }
    Write-Output ('LOCAL CHECKS FINISHED. No real client started. Private logs: ' + $privateDir)
} finally {
    Pop-Location
    foreach ($name in $names) { [Environment]::SetEnvironmentVariable($name, $saved[$name], 'Process') }
}
