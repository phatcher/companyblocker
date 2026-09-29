param(
    [switch]$NoRelease,
    [switch]$UseVsDevCmd,
    [string]$VsDevCmdPath = "C:\Program Files\Microsoft Visual Studio\18\Enterprise\Common7\Tools\VsDevCmd.bat",
    [string]$BinaryPath = "tools/bin/wikisieve.exe",
    [switch]$SkipBuild,
    [switch]$DryRun
)

$repoRoot = Split-Path -Parent $PSScriptRoot
$manifestDir = Join-Path $repoRoot "src/rust/wikisieve"
$profileDir = if ($NoRelease) { "debug" } else { "release" }
$builtBinaryPath = Join-Path $manifestDir "target/$profileDir/wikisieve.exe"
$deployedBinaryPath = if ([System.IO.Path]::IsPathRooted($BinaryPath)) { $BinaryPath } else { Join-Path $repoRoot $BinaryPath }

if (-not $SkipBuild) {
    $buildScriptPath = Join-Path $PSScriptRoot "build_wikisieve.ps1"
    $buildArgs = @{}
    if ($NoRelease) { $buildArgs.NoRelease = $true }
    if ($UseVsDevCmd) { $buildArgs.UseVsDevCmd = $true }
    if (-not [string]::IsNullOrWhiteSpace($VsDevCmdPath)) { $buildArgs.VsDevCmdPath = $VsDevCmdPath }
    if ($DryRun) { $buildArgs.DryRun = $true }

    & $buildScriptPath @buildArgs
    if ($LASTEXITCODE -ne 0) {
        throw "wikisieve build helper failed with exit code $LASTEXITCODE"
    }
}

Write-Host "[wikisieve-deploy] source: $builtBinaryPath"
Write-Host "[wikisieve-deploy] target: $deployedBinaryPath"

if ($DryRun) {
    Write-Host "[wikisieve-deploy] dry-run copy skipped"
}
else {
    if (-not (Test-Path $builtBinaryPath)) {
        throw "Built wikisieve binary not found: $builtBinaryPath"
    }
    $deployedParent = Split-Path -Parent $deployedBinaryPath
    if ($deployedParent) {
        New-Item -ItemType Directory -Force -Path $deployedParent | Out-Null
    }
    Copy-Item -LiteralPath $builtBinaryPath -Destination $deployedBinaryPath -Force

    # Read back the version line straight from the binary just deployed, rather than
    # asking git separately, so this always names the commit that binary itself reports
    # -- the one thing a hand copy never recorded.
    $deployedVersion = & $deployedBinaryPath --version
    Write-Host "[wikisieve-deploy] version: $deployedVersion"
}

Write-Host "[wikisieve-deploy] done"
