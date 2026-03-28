# Quick start script for Vertex AI training
# This script walks you through the entire setup process

param(
    [string]$ProjectId = "",
    [string]$Region = "us-central1",
    [string]$BucketName = ""
)

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Vertex AI Training - Quick Start" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Verify GCP CLI
Write-Host "[Step 1] Verifying Google Cloud CLI..." -ForegroundColor Yellow
try {
    $version = gcloud --version 2>&1 | Select-Object -First 1
    Write-Host "✓ Google Cloud SDK installed: $version" -ForegroundColor Green
}
catch {
    Write-Host "✗ Google Cloud SDK not found. Install from:" -ForegroundColor Red
    Write-Host "  https://cloud.google.com/sdk/docs/install" -ForegroundColor Red
    exit 1
}

# Step 2: Authenticate
Write-Host ""
Write-Host "[Step 2] Checking authentication..." -ForegroundColor Yellow
$authStatus = gcloud auth list --filter=status:ACTIVE --format="value(account)" 2>&1
if ($authStatus) {
    Write-Host "✓ Authenticated as: $authStatus" -ForegroundColor Green
}
else {
    Write-Host "⚠ Not authenticated. Running: gcloud auth login" -ForegroundColor Yellow
    gcloud auth login
}

# Step 3: Get Project ID
Write-Host ""
Write-Host "[Step 3] Setting project..." -ForegroundColor Yellow
if (-not $ProjectId) {
    $ProjectId = gcloud config get-value project 2>&1
    Write-Host "Current project: $ProjectId" -ForegroundColor Cyan
    $newProject = Read-Host "Enter your GCP Project ID (or press Enter to use current)"
    if ($newProject) {
        $ProjectId = $newProject
    }
}

gcloud config set project $ProjectId
Write-Host "✓ Project set to: $ProjectId" -ForegroundColor Green

# Step 4: Enable Services
Write-Host ""
Write-Host "[Step 4] Enabling required services (this may take 2-3 minutes)..." -ForegroundColor Yellow
Write-Host "  Enabling: aiplatform.googleapis.com"
gcloud services enable aiplatform.googleapis.com --quiet

Write-Host "  Enabling: artifactregistry.googleapis.com"
gcloud services enable artifactregistry.googleapis.com --quiet

Write-Host "  Enabling: cloudbuild.googleapis.com"
gcloud services enable cloudbuild.googleapis.com --quiet

Write-Host "  Enabling: storage-api.googleapis.com"
gcloud services enable storage-api.googleapis.com --quiet

Write-Host "✓ Services enabled" -ForegroundColor Green

# Step 5: Create GCS Bucket
Write-Host ""
Write-Host "[Step 5] Setting up Cloud Storage..." -ForegroundColor Yellow

if (-not $BucketName) {
    $randomNum = Get-Random -Minimum 10000 -Maximum 99999
    $BucketName = "ocr-data-$randomNum"
}

Write-Host "Creating bucket: $BucketName"
try {
    gsutil mb -p $ProjectId -l $Region gs://$BucketName 2>&1 | Where-Object { $_ -notlike "ServiceException: 409*" }
    Write-Host "✓ Bucket created/verified: gs://$BucketName" -ForegroundColor Green
}
catch {
    if ($_ -like "*409*") {
        Write-Host "✓ Bucket already exists: gs://$BucketName" -ForegroundColor Green
    }
    else {
        Write-Host "✗ Error creating bucket: $_" -ForegroundColor Red
        exit 1
    }
}

# Step 6: Upload Data
Write-Host ""
Write-Host "[Step 6] Uploading training data..." -ForegroundColor Yellow

$dataDir = "data\difficulty_labels"
if (-not (Test-Path $dataDir)) {
    Write-Host "✗ Data directory not found: $dataDir" -ForegroundColor Red
    exit 1
}

Write-Host "Uploading classifier data to gs://$BucketName/data/"
gsutil -m cp -r $dataDir gs://$BucketName/data/

Write-Host "✓ Data uploaded" -ForegroundColor Green

# Step 7: Create Artifact Registry
Write-Host ""
Write-Host "[Step 7] Setting up Artifact Registry..." -ForegroundColor Yellow
Write-Host "Creating repository..."

try {
    gcloud artifacts repositories create vertex-ai `
        --repository-format=docker `
        --location=$Region `
        --project=$ProjectId 2>&1 | Out-Null
    Write-Host "✓ Artifact Registry repository created" -ForegroundColor Green
}
catch {
    if ($_ -like "*already exists*") {
        Write-Host "✓ Artifact Registry repository already exists" -ForegroundColor Green
    }
    else {
        Write-Host "⚠ Note: $_ (continuing...)" -ForegroundColor Yellow
    }
}

# Step 8: Build Docker Image
Write-Host ""
Write-Host "[Step 8] Building Docker image..." -ForegroundColor Yellow

$imageName = "ocr-classifier-training"
$imageTag = "latest"
$imageUri = "$Region-docker.pkg.dev/$ProjectId/vertex-ai/${imageName}:$imageTag"

Write-Host "Building: $imageUri"
Write-Host "(This may take 10-15 minutes...)" -ForegroundColor Cyan

gcloud builds submit `
    --tag=$imageUri `
    --dockerfile=Dockerfile.training `
    --region=$Region

if ($LASTEXITCODE -ne 0) {
    Write-Host "✗ Docker build failed" -ForegroundColor Red
    exit 1
}

Write-Host "✓ Docker image built and pushed" -ForegroundColor Green

# Step 9: Summary and Next Steps
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "✓ Setup Complete!" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Your configuration:" -ForegroundColor Yellow
Write-Host "  Project ID:     $ProjectId"
Write-Host "  Region:         $Region"
Write-Host "  Bucket:         gs://$BucketName"
Write-Host "  Docker Image:   $imageUri"
Write-Host ""
Write-Host "Next: Submit training job with:" -ForegroundColor Yellow
Write-Host ""
Write-Host "python scripts/submit_vertex_training.py \" -ForegroundColor Cyan
Write-Host "    --project-id $ProjectId \" -ForegroundColor Cyan
Write-Host "    --region $Region \" -ForegroundColor Cyan
Write-Host "    --bucket-name $BucketName \" -ForegroundColor Cyan
Write-Host "    --image-uri $imageUri" -ForegroundColor Cyan
Write-Host ""
Write-Host "For detailed guide, see: VERTEX_AI_BEGINNER_GUIDE.md" -ForegroundColor Green
Write-Host ""
