# Build, test or lint the wikisieve crate through cargo.
#
# cargo needs a complete MSVC toolchain. On a machine with a partial Visual Studio 18 Insiders
# install, the default linker resolution picks it up and fails with
# `LINK : fatal error LNK1104: cannot open file 'msvcrt.lib'`. -UseVsDevCmd loads the Enterprise
# edition's environment instead. To run cargo directly, set the linker environment by hand,
# adjusting the MSVC and SDK versions to whatever `VC\Tools\MSVC` and `Windows Kits\10\Lib` hold:
#
#   $vc = "C:\Program Files\Microsoft Visual Studio\18\Enterprise\VC\Tools\MSVC\14.51.36231"
#   $sdk = "C:\Program Files (x86)\Windows Kits\10"; $sdkver = "10.0.26100.0"
#   $env:INCLUDE = "$vc\include;$sdk\Include\$sdkver\ucrt;$sdk\Include\$sdkver\shared;$sdk\Include\$sdkver\um;$sdk\Include\$sdkver\winrt"
#   $env:LIB = "$vc\lib\x64;$sdk\Lib\$sdkver\ucrt\x64;$sdk\Lib\$sdkver\um\x64"
#   $env:PATH = "$vc\bin\HostX64\x64;$env:PATH"
param(
    [string]$CargoSubcommand = "build",
    [string[]]$CargoExtraArgs,
    [string]$CargoExtraArgsLine,
    [switch]$NoRelease,
    [switch]$UseVsDevCmd,
    [string]$VsDevCmdPath = "C:\Program Files\Microsoft Visual Studio\18\Enterprise\Common7\Tools\VsDevCmd.bat",
    [switch]$SkipVenvActivation,
    [switch]$DryRun
)

$repoRoot = Split-Path -Parent $PSScriptRoot
$rustProjectPath = Join-Path $repoRoot "src/rust/wikisieve"
$manifestPath = Join-Path $rustProjectPath "Cargo.toml"

if (-not (Test-Path $manifestPath)) {
    throw "Rust manifest not found: $manifestPath"
}

$cargoArgs = @($CargoSubcommand)
if ($CargoSubcommand -eq "build" -and -not $NoRelease) {
    $cargoArgs += "--release"
}
if ($CargoExtraArgs) {
    $cargoArgs += $CargoExtraArgs
}
if (-not [string]::IsNullOrWhiteSpace($CargoExtraArgsLine)) {
    $cargoArgs += $CargoExtraArgsLine.Split(' ', [System.StringSplitOptions]::RemoveEmptyEntries)
}

Write-Host "[wikisieve-build] project:  $rustProjectPath"
Write-Host "[wikisieve-build] command:  cargo $($cargoArgs -join ' ')"
if ($CargoSubcommand -eq "build") {
    Write-Host "[wikisieve-build] profile:  $(if ($NoRelease) { 'debug' } else { 'release' })"
}

Push-Location $rustProjectPath
try {
    if (-not $SkipVenvActivation) {
        $venvActivatePath = Join-Path $repoRoot ".venv\Scripts\Activate.ps1"
        if (Test-Path $venvActivatePath) {
            . $venvActivatePath
        }
    }

    if ($UseVsDevCmd) {
        if (-not (Test-Path $VsDevCmdPath)) {
            throw "VsDevCmd.bat not found: $VsDevCmdPath"
        }

            if ($DryRun) {
                Write-Host "[wikisieve-build] dry-run env bootstrap: $VsDevCmdPath -arch=x64 -host_arch=x64"
                Write-Host "[wikisieve-build] dry-run command: cargo $($cargoArgs -join ' ')"
            }
            else {
                $cmdLine = "call ""$VsDevCmdPath"" -arch=x64 -host_arch=x64 && set"
                $envDump = cmd.exe /c $cmdLine
                if ($LASTEXITCODE -ne 0) {
                    throw "VsDevCmd bootstrap failed with exit code $LASTEXITCODE"
                }

                foreach ($line in $envDump) {
                    if ($line -match '^(.*?)=(.*)$') {
                        [System.Environment]::SetEnvironmentVariable($matches[1], $matches[2], 'Process')
                    }
            }

                & cargo @cargoArgs
                if ($LASTEXITCODE -ne 0) {
                    throw "Rust build failed with exit code $LASTEXITCODE"
                }
        }
    }
    else {
        if ($DryRun) {
            Write-Host "[wikisieve-build] dry-run command: cargo $($cargoArgs -join ' ')"
        }
        else {
            & cargo @cargoArgs
            if ($LASTEXITCODE -ne 0) {
                throw "Rust build failed with exit code $LASTEXITCODE"
            }
        }
    }
}
finally {
    Pop-Location
}

Write-Host "[wikisieve-build] done"
