param(
    [string]$ProjectId = "ocr-training-491603",
    [string]$Zone = "us-central1-a",
    [string]$InstanceName = "ocr-train-ce-$(Get-Date -Format 'yyyyMMdd-HHmmss')",
    [string]$MachineType = "n2-highcpu-16",
    [string]$ImageUri = "us-central1-docker.pkg.dev/ocr-training-491603/vertex-ai/ocr-classifier-training:latest",
    [int]$BootDiskGb = 200,
    [int]$Epochs = 30,
    [int]$BatchSize = 32,
    [double]$GradClipNorm = 1.0,
    [ValidateSet("greedy", "beam")]
    [string]$MetricDecodeStrategy = "greedy",
    [int]$MetricBeamWidth = 10,
    [int]$ValMetricBatches = 5,
    [switch]$EnableLmPostCorrection,
    [ValidateSet("compound", "word")]
    [string]$LmPostCorrectionMode = "compound",
    [switch]$AutoDeleteOnStop
)

$ErrorActionPreference = "Stop"

$metadataItems = @()
if ($AutoDeleteOnStop) {
    $metadataItems += "shutdown-script=sudo gcloud compute instances delete $InstanceName --zone $Zone --quiet"
}

Write-Host "Creating Compute Engine instance and starting OCR training container..."
Write-Host "Project:      $ProjectId"
Write-Host "Zone:         $Zone"
Write-Host "Instance:     $InstanceName"
Write-Host "Machine type: $MachineType"
Write-Host "Image:        $ImageUri"
Write-Host ""

$createCmd = @(
    "gcloud", "compute", "instances", "create-with-container", $InstanceName,
    "--project", $ProjectId,
    "--zone", $Zone,
    "--machine-type", $MachineType,
    "--boot-disk-size", "${BootDiskGb}GB",
    "--scopes", "https://www.googleapis.com/auth/cloud-platform",
    "--container-image", $ImageUri,
    "--container-restart-policy", "never",
    "--container-command", "python",
    "--container-arg=-m",
    "--container-arg=src.ocr.custom_model.train",
    "--container-arg=--epochs",
    "--container-arg=$Epochs",
    "--container-arg=--batch-size",
    "--container-arg=$BatchSize",
    "--container-arg=--grad-clip-norm",
    "--container-arg=$GradClipNorm",
    "--container-arg=--metric-decode-strategy",
    "--container-arg=$MetricDecodeStrategy",
    "--container-arg=--metric-beam-width",
    "--container-arg=$MetricBeamWidth",
    "--container-arg=--val-metric-batches",
    "--container-arg=$ValMetricBatches",
    "--container-arg=--gcs-processed-prefix",
    "--container-arg=data/processed",
    "--container-arg=--gcs-raw-prefix",
    "--container-arg=data/raw/iam",
    "--container-mount-host-path", "mount-path=/app/models,host-path=/var/ocr-models,mode=rw",
    "--container-mount-host-path", "mount-path=/app/logs,host-path=/var/ocr-logs,mode=rw",
    "--format", "value(name,status)"
)

if ($EnableLmPostCorrection) {
    $createCmd += @(
        "--container-arg=--enable-lm-post-correction",
        "--container-arg=--lm-post-correction-mode",
        "--container-arg=$LmPostCorrectionMode"
    )
}

if ($metadataItems.Count -gt 0) {
    $createCmd += @("--metadata", ($metadataItems -join ","))
}

$createArgs = $createCmd[1..($createCmd.Length - 1)]
& $createCmd[0] @createArgs
if ($LASTEXITCODE -ne 0) {
    throw "Failed to create instance and start container."
}

Write-Host ""
Write-Host "Training started on Compute Engine."
Write-Host ""
Write-Host "Tail live logs:"
Write-Host "  gcloud compute ssh $InstanceName --project $ProjectId --zone $Zone --command \"sudo journalctl -u konlet-startup --no-pager -f\""
Write-Host ""
Write-Host "Check instance status:"
Write-Host "  gcloud compute instances describe $InstanceName --project $ProjectId --zone $Zone --format='get(status)'"
Write-Host ""
Write-Host "When training finishes, copy artifacts:"
Write-Host "  gcloud compute scp --recurse ${InstanceName}:/var/ocr-models ./models_from_compute --project $ProjectId --zone $Zone"
Write-Host "  gcloud compute scp --recurse ${InstanceName}:/var/ocr-logs ./logs_from_compute --project $ProjectId --zone $Zone"
Write-Host ""
Write-Host "Delete instance when done:"
Write-Host "  gcloud compute instances delete $InstanceName --project $ProjectId --zone $Zone --quiet"
