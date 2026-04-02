$ErrorActionPreference = "Stop"
$projectRoot = "C:\Users\brosi\Desktop\SEG4180\Project"
$tempDir = Join-Path $projectRoot ".gcloud-tmp"

$env:TEMP = $tempDir
$env:TMP = $tempDir
New-Item -ItemType Directory -Force $tempDir | Out-Null

Set-Location $projectRoot
gcloud builds submit . --config cloudbuild.training.yaml