param(
    [switch]$RefreshLock,
    [switch]$NoLock
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$lockFile = 'uv.lock'

Set-Location -Path $PSScriptRoot

if (Test-Path .venv) {
    Remove-Item -Recurse -Force .venv
}

if (Get-Command py -ErrorAction SilentlyContinue) {
    py -3.12 -m venv .venv
} else {
    python -m venv .venv
}

.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install uv==0.11.23

if ($RefreshLock) {
    & .\.venv\Scripts\uv.exe lock --upgrade
} elseif (-not (Test-Path $lockFile)) {
    & .\.venv\Scripts\uv.exe lock
}

if ((-not $NoLock) -and (Test-Path $lockFile)) {
    & .\.venv\Scripts\uv.exe sync --frozen --all-groups
} else {
    & .\.venv\Scripts\uv.exe sync --all-groups
}

Write-Host "Virtual environment recreated at .venv" -ForegroundColor Green
. .\.venv\Scripts\Activate.ps1
Write-Host "Virtual environment activated: $env:VIRTUAL_ENV" -ForegroundColor Green
