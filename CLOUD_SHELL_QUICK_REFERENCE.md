# Cloud Shell Quick Reference - Vertex AI Training

**Fastest way to start training on Vertex AI - no local setup needed!**

---

## Step 1: Open Cloud Shell

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Click the **>_** icon (top-right) to open Cloud Shell
3. Wait for terminal to appear at bottom

---

## Step 2: Run All-in-One Setup

Copy and paste this entire script into Cloud Shell:

```bash
#!/bin/bash

# ===== CONFIGURATION =====
export PROJECT_ID="ocr-training-12345"  # CHANGE THIS to your project ID
export REGION="us-central1"
export BUCKET_NAME="ocr-data-$(shuf -i 10000-99999 -n 1)"
export IMAGE_NAME="ocr-classifier-training"

echo "=========================================="
echo "Vertex AI Training - Cloud Shell Setup"
echo "=========================================="
echo "Project ID:    $PROJECT_ID"
echo "Region:        $REGION"
echo "Bucket:        $BUCKET_NAME"
echo "=========================================="
echo ""

# Step 1: Set project
echo "[1/6] Setting project..."
gcloud config set project $PROJECT_ID

# Step 2: Enable services
echo "[2/6] Enabling services (takes ~2 min)..."
gcloud services enable aiplatform.googleapis.com
gcloud services enable artifactregistry.googleapis.com
gcloud services enable cloudbuild.googleapis.com
gcloud services enable storage-api.googleapis.com
echo "✓ Services enabled"

# Step 3: Create bucket
echo "[3/6] Creating Cloud Storage bucket..."
gsutil mb -p $PROJECT_ID -l $REGION gs://$BUCKET_NAME
echo "✓ Bucket created: gs://$BUCKET_NAME"

# Step 4: Create Artifact Registry
echo "[4/6] Creating Artifact Registry..."
gcloud artifacts repositories create vertex-ai \
    --repository-format=docker \
    --location=$REGION 2>/dev/null || echo "✓ Artifact Registry already exists"
echo "✓ Artifact Registry ready"

# Step 5: Build Docker image
echo "[5/6] Building Docker image..."
echo "  This may take 10-15 minutes..."
IMAGE_URI="$REGION-docker.pkg.dev/$PROJECT_ID/vertex-ai/$IMAGE_NAME:latest"

gcloud builds submit \
    --tag=$IMAGE_URI \
    --dockerfile=Dockerfile.training \
    --region=$REGION

if [ $? -ne 0 ]; then
    echo "✗ Docker build failed"
    exit 1
fi

echo "✓ Docker image ready"

# Step 6: Summary
echo ""
echo "=========================================="
echo "✓ Setup Complete!"
echo "=========================================="
echo ""
echo "Your configuration:"
echo "  Project ID:    $PROJECT_ID"
echo "  Bucket:        gs://$BUCKET_NAME"
echo "  Docker Image:  $IMAGE_URI"
echo ""
echo "Next steps:"
echo "  1. Upload data using Cloud Shell upload button (⤓)"
echo "  2. Copy data to GCS:"
echo ""
echo "gsutil -m cp -r data/difficulty_labels gs://$BUCKET_NAME/data/"
echo "gsutil -m cp -r data/processed gs://$BUCKET_NAME/data/"
echo ""
echo "  3. Submit training job:"
echo ""
echo "pip install google-cloud-aiplatform google-cloud-storage"
echo ""
echo "python scripts/submit_vertex_training.py \\"
echo "    --project-id $PROJECT_ID \\"
echo "    --region $REGION \\"
echo "    --bucket-name $BUCKET_NAME \\"
echo "    --image-uri $IMAGE_URI"
echo ""
echo "=========================================="
```

---

## Step 3: Upload Your Code & Data

After the script completes, upload your files:

1. Click the **⤓** (upload) button at top-right of Cloud Shell
2. Select your project folder (containing Dockerfile.training, src/, scripts/, data/, requirements.txt)
3. Wait for upload to complete

Then extract it:
```bash
unzip project.zip
cd SEG4180/Project
```

---

## Step 4: Copy Data to Cloud Storage

```bash
# Set your bucket name (from setup output above)
export BUCKET_NAME="ocr-data-12345"  # Replace with your actual bucket

# Copy training data
gsutil -m cp -r data/difficulty_labels gs://$BUCKET_NAME/data/
gsutil -m cp -r data/processed gs://$BUCKET_NAME/data/

# Verify
gsutil ls -r gs://$BUCKET_NAME/data/ | head -20
```

---

## Step 5: Submit Training Job

```bash
# Install GCP Python libraries
pip install google-cloud-aiplatform google-cloud-storage

# Set variables (from setup output)
export PROJECT_ID="ocr-training-12345"
export BUCKET_NAME="ocr-data-12345"
export REGION="us-central1"
export IMAGE_URI="us-central1-docker.pkg.dev/$PROJECT_ID/vertex-ai/ocr-classifier-training:latest"

# Run submission script
python scripts/submit_vertex_training.py \
    --project-id $PROJECT_ID \
    --region $REGION \
    --bucket-name $BUCKET_NAME \
    --image-uri $IMAGE_URI
```

**You should see:**
```
================================================================================
✓ Job submitted successfully!

Job Resource Name:
  projects/123456789/locations/us-central1/trainingPipelines/1234567890

Monitor your training job:
  Web UI: https://console.cloud.google.com/vertex-ai/training/custom-jobs?
          project=ocr-training-12345
================================================================================
```

---

## Step 6: Monitor Training

### In Cloud Shell:
```bash
# List all training jobs
gcloud ai custom-jobs list --region=us-central1

# Stream live logs (replace with your job ID)
gcloud ai custom-jobs stream-logs TRAINING_PIPELINE_ID --region=us-central1
```

### In Web UI:
1. Click the link from the job submission output
2. Or go to: https://console.cloud.google.com/vertex-ai/training/custom-jobs

---

## Common Cloud Shell Commands

```bash
# File management
ls -la                          # List files
cd data/                        # Change directory
cat config.yaml                 # View file
mkdir new_folder                # Create folder
rm filename                      # Delete file
mv oldname newname              # Rename

# Working with GCS
gsutil ls                        # List all buckets
gsutil ls gs://bucket-name/     # List bucket contents
gsutil cp file.txt gs://bucket/ # Copy to GCS
gsutil -m cp -r folder/ gs://bucket/  # Copy folder (multi-threaded)

# Working with Vertex AI
gcloud ai custom-jobs list --region=us-central1  # List jobs
gcloud ai models list --region=us-central1        # List models

# Python in Cloud Shell
python --version                # Check Python version
pip list                        # List installed packages
python script.py               # Run script
```

---

## Troubleshooting in Cloud Shell

**Problem: "Permission denied" on upload**
```bash
# Make sure you're in the right directory
pwd  # Shows current path
```

**Problem: Files disappeared**
```bash
# Cloud Shell has a persistent home directory
# Files in home (~/) are kept
# Files in /tmp/ are deleted
ls ~/                          # Check home directory
```

**Problem: Out of disk space**
```bash
# Cloud Shell has 5GB persistent storage
# Check usage
du -sh ~/*
```

**Problem: Need to stop a training job**
```bash
# Cancel from web UI: https://console.cloud.google.com/vertex-ai/training/custom-jobs
# Or list and note the ID
gcloud ai custom-jobs list --region=us-central1

# Job cannot be cancelled via CLI, use web UI
```

---

## Cost Monitor

Check your running costs in Cloud Shell:

```bash
# View billing account
gcloud billing accounts list

# View costs for this project
gcloud billing projects describe $PROJECT_ID
```

Or in web UI:
- Go to: https://console.cloud.google.com/billing
- Select your project
- View current month charges

---

## Next: Download Results

When training completes, download your models:

```bash
# Set bucket name
export BUCKET_NAME="ocr-data-12345"

# Download entire models folder
gsutil -m cp -r gs://$BUCKET_NAME/models/classifier ~/downloaded_models/

# Or download just the best model
gsutil cp gs://$BUCKET_NAME/models/classifier/best_model.keras ~/best_model.keras
```

Then download from Cloud Shell to your computer:
1. Click **⋮** menu (top-right of terminal)
2. Select "Download file"
3. Navigate and select files

---

## Cloud Shell Limitations & Workarounds

| Limitation | Workaround |
|-----------|-----------|
| Session expires after 1 hour of inactivity | Just reconnect - files are saved |
| 5GB persistent disk | Use GCS for large data |
| Limited GPU for local work | That's what Vertex AI is for! |
| Can't edit files with GUI | Use `nano` or `vim` in terminal |

---

## Tips & Tricks

**Bookmark this command:**
```bash
# Quick reference for checking job status
function check_training() {
    PROJECT=$1
    echo "Recent training jobs:"
    gcloud ai custom-jobs list --region=us-central1 --project=$PROJECT --limit=5
}

check_training ocr-training-12345
```

**Use aliases for faster commands:**
```bash
# In Cloud Shell, create shortcuts
alias myproject="gcloud config set project ocr-training-12345"
alias listjobs="gcloud ai custom-jobs list --region=us-central1"
alias mybucket="gsutil ls gs://ocr-data-12345/"
```

---

## Quick FAQ

**Q: Where are my files stored in Cloud Shell?**  
A: Home directory (`~`) is persistent. Cloud Shell also has access to GCS buckets.

**Q: Can I edit code in Cloud Shell?**  
A: Yes! Use `nano filename` to edit, or `code .` for VS Code integration.

**Q: How long can my training job run?**  
A: Up to 7 days for Vertex AI Training.

**Q: Can I stop and resume training?**  
A: Most training jobs can be stopped, but TensorFlow training must restart from checkpoint (saved models can be downloaded).

---

**Happy training! 🚀**
