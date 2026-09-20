param(
    [Parameter(Mandatory=$true)]
    [string]$TargetRepo
)

$ErrorActionPreference = "Stop"

$SourceRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$TargetRepo = (Resolve-Path $TargetRepo).Path

$BackendDir = Join-Path $TargetRepo "backend\app\voice_detector"
if (-not (Test-Path $BackendDir)) {
    throw "Target does not look like Vivansh-07/VaaniRakshak-AI: missing $BackendDir"
}

$AdapterSource = Join-Path $SourceRoot "backend\app\voice_detector\wavlm_v2.py"
$LauncherSource = Join-Path $SourceRoot "run_v2_prototype.py"

Copy-Item $AdapterSource (Join-Path $BackendDir "wavlm_v2.py") -Force
Copy-Item $LauncherSource (Join-Path $TargetRepo "run_v2_prototype.py") -Force

Write-Host ""
Write-Host "V2 backend bridge installed." -ForegroundColor Green
Write-Host "  adapter : backend\app\voice_detector\wavlm_v2.py"
Write-Host "  launcher: run_v2_prototype.py"
Write-Host ""
Write-Host "Next:"
Write-Host "  cd $TargetRepo"
Write-Host "  python -m pip install transformers"
Write-Host "  python run_v2_prototype.py --checkpoint C:\VaaniRakshakData\models\v2_wavlm_mlaad_tiny\best.pt --device cuda --stt none --open"
