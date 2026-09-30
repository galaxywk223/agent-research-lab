param(
  [int]$BackendPort = 2024,
  [int]$FrontendPort = 3000,
  [switch]$OpenBrowser
)

$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
$root = Split-Path -Parent $PSScriptRoot
$frontendRoot = Join-Path $root "frontend"
$runtimeRoot = Join-Path $root ".runtime"
$startupLog = Join-Path $runtimeRoot "startup.log"
$backendStdoutLog = Join-Path $runtimeRoot "backend.stdout.log"
$backendStderrLog = Join-Path $runtimeRoot "backend.stderr.log"
$frontendStdoutLog = Join-Path $runtimeRoot "frontend.stdout.log"
$frontendStderrLog = Join-Path $runtimeRoot "frontend.stderr.log"
New-Item -ItemType Directory -Path $runtimeRoot -Force | Out-Null

function Write-StartupLog([string]$Message) {
  $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
  Add-Content -LiteralPath $startupLog -Value "[$timestamp] $Message"
}

function Fail([string]$Message) {
  Write-StartupLog "ERROR: $Message"
  Write-Error $Message
  exit 1
}

function Test-PortListening([int]$Port) {
  try {
    $connection = Get-NetTCPConnection -LocalAddress "127.0.0.1" -LocalPort $Port -State Listen -ErrorAction Stop
    return $null -ne $connection
  } catch {
    return $false
  }
}

function Get-PortProcesses([int]$Port) {
  try {
    $connections = @(Get-NetTCPConnection -LocalAddress "127.0.0.1" -LocalPort $Port -State Listen -ErrorAction Stop)
    foreach ($connection in $connections) {
      Get-CimInstance Win32_Process -Filter "ProcessId=$($connection.OwningProcess)" -ErrorAction SilentlyContinue
    }
  } catch {
    return @()
  }
}

function Test-ProjectProcessOnPort([int]$Port, [string]$ProjectMarker) {
  $marker = [IO.Path]::GetFullPath($ProjectMarker).TrimEnd('\')
  foreach ($process in @(Get-PortProcesses $Port)) {
    if ($process.CommandLine -and $process.CommandLine.IndexOf($marker, [StringComparison]::OrdinalIgnoreCase) -ge 0) {
      return $true
    }
  }
  return $false
}

function Wait-ForPorts([int[]]$Ports, [int]$TimeoutSeconds = 60) {
  for ($second = 0; $second -lt $TimeoutSeconds; $second++) {
    $allReady = $true
    foreach ($port in $Ports) {
      if (-not (Test-PortListening $port)) {
        $allReady = $false
        break
      }
    }
    if ($allReady) {
      return $true
    }
    Start-Sleep -Seconds 1
  }
  return $false
}

function Get-EnvValue([string]$Path, [string]$Name) {
  $line = Get-Content -LiteralPath $Path | Where-Object {
    $_ -match "^\s*$([regex]::Escape($Name))\s*="
  } | Select-Object -First 1
  if ($null -eq $line) {
    return $null
  }
  return (($line -split "=", 2)[1]).Trim().Trim('"').Trim("'")
}

if ($BackendPort -lt 1 -or $BackendPort -gt 65535) {
  Fail "BackendPort must be between 1 and 65535."
}
if ($FrontendPort -lt 1 -or $FrontendPort -gt 65535) {
  Fail "FrontendPort must be between 1 and 65535."
}
if ($BackendPort -eq $FrontendPort) {
  Fail "BackendPort and FrontendPort must be different."
}

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
  Fail "uv was not found. Install uv and ensure it is available in PATH."
}
if (-not (Get-Command pnpm -ErrorAction SilentlyContinue)) {
  Fail "pnpm was not found. Install pnpm and ensure it is available in PATH."
}
$envPath = Join-Path $root ".env"
$envExamplePath = Join-Path $root ".env.example"
if (-not (Test-Path -LiteralPath $envPath)) {
  if (-not (Test-Path -LiteralPath $envExamplePath)) {
    Fail "Missing .env.example. Restore the project template before starting."
  }
  Copy-Item -LiteralPath $envExamplePath -Destination $envPath
  Start-Process -FilePath "notepad.exe" -ArgumentList "`"$envPath`""
  Write-Host "Created .env from .env.example and opened it in Notepad."
  Write-Host "Fill in the required model and search settings, save the file, then run the startup script again."
  exit 2
}

if (-not (Test-Path -LiteralPath (Join-Path $frontendRoot "node_modules\.bin\next.cmd"))) {
  Fail "Frontend dependencies are missing. Run 'pnpm install --frozen-lockfile' in the frontend directory."
}

$modelNames = @(
  "RESEARCH_MODEL",
  "SUMMARIZATION_MODEL",
  "COMPRESSION_MODEL",
  "FINAL_REPORT_MODEL"
) | ForEach-Object { Get-EnvValue $envPath $_ }
$requiredKeys = @()
foreach ($modelName in $modelNames) {
  if ([string]::IsNullOrWhiteSpace($modelName)) {
    continue
  }
  if ($modelName -match "^openai:") {
    $requiredKeys += "OPENAI_API_KEY", "OPENAI_BASE_URL"
  } elseif ($modelName -match "^anthropic:") {
    $requiredKeys += "ANTHROPIC_API_KEY"
  } elseif ($modelName -match "^google") {
    $requiredKeys += "GOOGLE_API_KEY"
  } elseif ($modelName -match "^deepseek:") {
    $requiredKeys += "DEEPSEEK_API_KEY"
  }
}

$searchApi = (Get-EnvValue $envPath "SEARCH_API").ToLowerInvariant()
if ($searchApi -eq "tavily") {
  $requiredKeys += "TAVILY_API_KEY"
}
$missingKeys = @($requiredKeys | Select-Object -Unique | Where-Object {
  [string]::IsNullOrWhiteSpace((Get-EnvValue $envPath $_))
})
if ($missingKeys.Count -gt 0) {
  Start-Process -FilePath "notepad.exe" -ArgumentList "`"$envPath`""
  Fail "Missing required settings in .env: $($missingKeys -join ', '). Fill them in, save the file, and run the startup script again."
}

$tracingValue = Get-EnvValue $envPath "LANGSMITH_TRACING"
if ([string]::IsNullOrWhiteSpace($tracingValue)) {
  $env:LANGSMITH_TRACING = "false"
  Write-StartupLog "LANGSMITH_TRACING was empty; using false for this launch."
} elseif ($tracingValue -notmatch "^(?i:true|false)$") {
  Fail "LANGSMITH_TRACING must be true or false."
}

$backendRunning = Test-PortListening $BackendPort
$frontendRunning = Test-PortListening $FrontendPort
if ($backendRunning -and -not (Test-ProjectProcessOnPort $BackendPort $root)) {
  Fail "Backend port $BackendPort is already in use by another process."
}
if ($frontendRunning -and -not (Test-ProjectProcessOnPort $FrontendPort $frontendRoot)) {
  Fail "Frontend port $FrontendPort is already in use by another process."
}

$rootCommandPath = $root.Replace("'", "''")
$frontendCommandPath = $frontendRoot.Replace("'", "''")
if ($backendRunning) {
  Write-StartupLog "Reusing the existing project backend on port $BackendPort."
} else {
  foreach ($logPath in @($backendStdoutLog, $backendStderrLog)) {
    if (Test-Path -LiteralPath $logPath) { Remove-Item -LiteralPath $logPath -Force }
  }
  Write-StartupLog "Starting backend on port $BackendPort."
  Start-Process -FilePath "powershell.exe" -WorkingDirectory $root -WindowStyle Hidden `
    -RedirectStandardOutput $backendStdoutLog -RedirectStandardError $backendStderrLog -ArgumentList @(
      "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command",
      "Set-Location -LiteralPath '$rootCommandPath'; uv run --project . langgraph dev --no-browser --no-reload --port $BackendPort"
    )
}
if ($frontendRunning) {
  Write-StartupLog "Reusing the existing project frontend on port $FrontendPort."
} else {
  foreach ($logPath in @($frontendStdoutLog, $frontendStderrLog)) {
    if (Test-Path -LiteralPath $logPath) { Remove-Item -LiteralPath $logPath -Force }
  }
  Write-StartupLog "Starting frontend on port $FrontendPort."
  Start-Process -FilePath "powershell.exe" -WorkingDirectory $frontendRoot -WindowStyle Hidden `
    -RedirectStandardOutput $frontendStdoutLog -RedirectStandardError $frontendStderrLog -ArgumentList @(
      "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command",
      "Set-Location -LiteralPath '$frontendCommandPath'; pnpm dev --hostname 127.0.0.1 --port $FrontendPort"
    )
}

Write-Host "Backend: http://127.0.0.1:$BackendPort"
Write-Host "Frontend: http://127.0.0.1:$FrontendPort"

if ($OpenBrowser) {
  Write-Host "Waiting for backend and frontend services..."
  if (Wait-ForPorts @($BackendPort, $FrontendPort)) {
    Write-StartupLog "Backend and frontend are ready. Opening the frontend."
    Start-Process "http://127.0.0.1:$FrontendPort"
  } else {
    Fail "Services did not both become ready within 60 seconds. Check .runtime startup and service logs."
  }
}
