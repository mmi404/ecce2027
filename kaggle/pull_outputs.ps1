# Pulls every WP3 train/eval kernel's output down from Kaggle and routes the
# files into the project's results/ layout, regardless of how the Kaggle CLI
# nests them under the staging directory.
#
# Prereq: `kaggle auth login` (or an API token) must already be configured for
# this machine -- run that yourself first; this script never touches credentials.

$ErrorActionPreference = "Stop"

$Owner   = "mmi404"
$RunIds  = @("e1_0", "e1_1", "e1_2", "e2_0", "e2_1", "e2_2", "rob_a", "rob_b")
$Root    = "E:\41\ecce2027-ctg-transfer"
$Stage   = Join-Path $Root "results\_kaggle_raw"
$PredDir = Join-Path $Root "results\predictions"
$LogDir  = Join-Path $Root "results\logs"
$WgtDir  = Join-Path $Root "results\weights"

foreach ($d in @($Stage, $PredDir, $LogDir, $WgtDir)) {
    New-Item -ItemType Directory -Force -Path $d | Out-Null
}

$kernels = @()
foreach ($id in $RunIds) {
    $kernels += "train_run_$id"
    $kernels += "eval_run_$id"
}

$summary = @()

foreach ($nb in $kernels) {
    $slug = $nb -replace "_", "-"
    $dest = Join-Path $Stage $slug
    New-Item -ItemType Directory -Force -Path $dest | Out-Null

    Write-Host "=== Downloading $Owner/$slug ===" -ForegroundColor Cyan
    try {
        py -m kaggle kernels output "$Owner/$slug" -p $dest
    } catch {
        Write-Host "FAILED to download $slug -- $_" -ForegroundColor Red
        $summary += [PSCustomObject]@{ Kernel = $slug; Status = "DOWNLOAD FAILED" }
        continue
    }

    $files = Get-ChildItem -Path $dest -Recurse -File -ErrorAction SilentlyContinue
    $routed = 0
    foreach ($f in $files) {
        if ($f.Name -match "_preds_conf0001\.json$") {
            Copy-Item $f.FullName -Destination $PredDir -Force; $routed++
        } elseif ($f.Name -match "_eval_summary\.json$|_run_info\.json$|_results\.csv$|_args\.yaml$") {
            Copy-Item $f.FullName -Destination $LogDir -Force; $routed++
        } elseif ($f.Name -match "_last\.pt$") {
            Copy-Item $f.FullName -Destination $WgtDir -Force; $routed++
        }
    }
    Write-Host "  -> $($files.Count) files downloaded, $routed routed into results/." -ForegroundColor Green
    $summary += [PSCustomObject]@{ Kernel = $slug; Status = "OK"; FilesFound = $files.Count; Routed = $routed }
}

Write-Host "`n=== Summary ===" -ForegroundColor Yellow
$summary | Format-Table -AutoSize

Write-Host "`nPredictions: $((Get-ChildItem $PredDir -File -ErrorAction SilentlyContinue).Count) files in $PredDir"
Write-Host "Logs:        $((Get-ChildItem $LogDir -File -ErrorAction SilentlyContinue).Count) files in $LogDir"
Write-Host "Weights:     $((Get-ChildItem $WgtDir -File -ErrorAction SilentlyContinue).Count) files in $WgtDir"
