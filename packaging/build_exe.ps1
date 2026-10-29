# Build a Windows onedir EXE for this project.
# Reuses the existing Python 3.14 + torch 2.9.1+cu128 install.
# Run:
#   powershell -NoProfile -ExecutionPolicy Bypass -File packaging\build_exe.ps1
$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $Root

$BasePy = Join-Path $env:LOCALAPPDATA "Python\pythoncore-3.14-64\python.exe"
if (-not (Test-Path -LiteralPath $BasePy)) {
    throw "Python 3.14 not found: $BasePy"
}

$VenvPy = Join-Path $Root ".venv-build\Scripts\python.exe"
$VenvPi = Join-Path $Root ".venv-build\Scripts\pyinstaller.exe"

if (-not (Test-Path -LiteralPath $VenvPy)) {
    Write-Host "Creating build venv with --system-site-packages"
    & $BasePy -m venv --system-site-packages (Join-Path $Root ".venv-build")
}

Write-Host "Installing PyInstaller"
& $VenvPy -m pip install --upgrade pip pyinstaller

# Short work path: torch's tree exceeds Windows MAX_PATH under the Desktop folder.
$Out = Join-Path $Root "release\StereoCamera"
$Dist = "C:\st_build\dist"
$Work = "C:\st_build\work"
New-Item -ItemType Directory -Force -Path $Dist, $Work | Out-Null
if (Test-Path -LiteralPath $Out) {
    Write-Host "Removing old release\StereoCamera"
    Remove-Item -LiteralPath $Out -Recurse -Force
}
$Built = Join-Path $Dist "StereoCamera"
if (Test-Path -LiteralPath $Built) {
    Remove-Item -LiteralPath $Built -Recurse -Force
}

Write-Host "Running PyInstaller (CUDA libs are about 4 GB; copying takes a while)"
& $VenvPi --noconfirm --clean `
    --distpath $Dist `
    --workpath $Work `
    (Join-Path $Root "packaging\StereoCamera.spec")
if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed, exit $LASTEXITCODE" }

Write-Host "Copying build into release\"
New-Item -ItemType Directory -Force -Path $Out | Out-Null
& robocopy $Built $Out /E /NFL /NDL /NJH /NJS /nc /ns /np
if ($LASTEXITCODE -ge 8) { throw "robocopy failed, exit $LASTEXITCODE" }
$global:LASTEXITCODE = 0

Write-Host "Placing editable config, calibration and weights next to the exe"
Copy-Item -LiteralPath (Join-Path $Root "config.yaml") -Destination (Join-Path $Out "config.yaml") -Force
Copy-Item -LiteralPath (Join-Path $Root "calibration") -Destination (Join-Path $Out "calibration") -Recurse -Force
Copy-Item -LiteralPath (Join-Path $Root "weights") -Destination (Join-Path $Out "weights") -Recurse -Force
$readme = Get-ChildItem -LiteralPath $PSScriptRoot -Filter "README*.md" | Select-Object -First 1
if ($null -eq $readme) { throw "README not found in packaging\" }
Copy-Item -LiteralPath $readme.FullName -Destination (Join-Path $Out $readme.Name) -Force
New-Item -ItemType Directory -Force -Path (Join-Path $Out "output") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $Out "logs") | Out-Null

Write-Host "DONE $Out\StereoCamera.exe"
