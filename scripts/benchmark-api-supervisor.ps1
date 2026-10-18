param(
    [string]$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path,
    [string]$ApiDirectory = '.runtime/dev-v2-api',
    [int]$Port = 2035
)
$ErrorActionPreference = 'Stop'
$runtime = Join-Path $ProjectRoot $ApiDirectory
$recordPath = Join-Path $runtime 'process.json'
$logs = Join-Path $runtime 'logs'
New-Item -ItemType Directory -Path $logs -Force | Out-Null
$stamp = [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssfffZ')
$stdoutPath = Join-Path $logs "api-$stamp.stdout.log"
$stderrPath = Join-Path $logs "api-$stamp.stderr.log"
function Save-ProcessRecord($value) {
    $temporary = "$recordPath.tmp"
    $value | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $temporary -Encoding utf8
    Move-Item -LiteralPath $temporary -Destination $recordPath -Force
}
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:AGENTLAB_PROVIDER_MODE = 'real'
$env:AGENTLAB_METRICS_DIR = Join-Path $ProjectRoot '.runtime/metrics-dev-v2'
$env:AGENTLAB_RUNTIME_EVENTS_DIR = Join-Path $ProjectRoot '.runtime/runtime_events'
$env:LANGGRAPH_LOCAL_AUTH = 'true'
$envFile = Join-Path $ProjectRoot '.env'
if (Test-Path -LiteralPath $envFile) {
    Get-Content -LiteralPath $envFile | ForEach-Object {
        if ($_ -match '^\s*([^#=][^=]*)=(.*)$') {
            $name = $matches[1].Trim()
            $value = $matches[2].Trim().Trim('"').Trim("'")
            [Environment]::SetEnvironmentVariable($name, $value, 'Process')
        }
    }
}
$databaseUri = $env:AGENTLAB_DATABASE_URI
if (-not $databaseUri) { $databaseUri = $env:DATABASE_URI }
if (-not $databaseUri) { $databaseUri = 'postgresql+asyncpg://agentlab_metrics@127.0.0.1:55433/agentlab_dev_v2' }
$env:AGENTLAB_DATABASE_URI = $databaseUri
$env:DATABASE_URI = $databaseUri
$uv = Join-Path $ProjectRoot '.runtime/debug-completion/tools/uv/uv.exe'
if (-not (Test-Path -LiteralPath $uv)) { throw 'Project uv executable is unavailable.' }
Add-Type -AssemblyName System.Net.Http
$healthClient = [System.Net.Http.HttpClient]::new()
try {
    $process = Start-Process -FilePath $uv -ArgumentList @(
        'run', '--project', $ProjectRoot, 'langgraph', 'dev', '--config',
        (Join-Path $runtime 'langgraph.json'), '--no-browser', '--no-reload',
        '--allow-blocking', '--port', "$Port"
    ) -WorkingDirectory $runtime -PassThru -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath
    $record = [ordered]@{
        state = 'running'; api_pid = $process.Id; started_at = [DateTime]::UtcNow.ToString('o')
        exit_code = $null; last_health_success_at = $null; detected_at = $null
        stdout_log = $stdoutPath; stderr_log = $stderrPath; auto_restart = $false
    }
    Save-ProcessRecord $record
    while (-not $process.HasExited) {
        try {
            $health = $healthClient.GetAsync("http://127.0.0.1:$Port/ok").GetAwaiter().GetResult()
            if ($health.IsSuccessStatusCode) {
                $record.last_health_success_at = [DateTime]::UtcNow.ToString('o')
                Save-ProcessRecord $record
            }
        } catch { }
        Start-Sleep -Seconds 2
        $process.Refresh()
    }
    $record.state = 'exited'
    $record.exit_code = $process.ExitCode
    $record.detected_at = [DateTime]::UtcNow.ToString('o')
    Save-ProcessRecord $record
} finally {
    $healthClient.Dispose()
}
