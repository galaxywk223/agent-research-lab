param([string]$TaskName = 'AgentResearchLab-Benchmark-API')
$ErrorActionPreference = 'Stop'
$script = Join-Path $PSScriptRoot 'benchmark-api-supervisor.ps1'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$action = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$script`"" -WorkingDirectory $projectRoot
$principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType Interactive -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero) -MultipleInstances IgnoreNew -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -DontStopOnIdleEnd
Register-ScheduledTask -TaskName $TaskName -Action $action -Principal $principal -Settings $settings -Description 'On-demand isolated owner for the local benchmark LangGraph API; no automatic restart.' -Force | Out-Null
Write-Output "Registered on-demand task: $TaskName"
Write-Output "Start with: Start-ScheduledTask -TaskName '$TaskName'"
