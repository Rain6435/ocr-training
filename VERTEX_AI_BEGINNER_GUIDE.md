# Vertex AI Training - Complete Beginner's Guide

This guide walks you through training your models on Google Cloud Vertex AI, step-by-step.

---

## What is Vertex AI?

**Vertex AI** is Google Cloud's managed machine learning platform. Instead of training on your local computer, you upload your code and data to Google Cloud, which handles:

- Running the training
- Managing GPUs/TPUs
- Storing results
- Tracking experiments

---

## Prerequisites Checklist

- [ ] Google Cloud account (create free: https://cloud.google.com/free)
- [ ] GitHub account with your project code pushed
- [ ] $300 free GCP credit (new accounts)

**That's it!** No local installation needed - everything runs in your browser.

---

## ⭐ GitHub + Cloud Shell Workflow

**This is the EASIEST approach!** Use GitHub to store your code, then train on Vertex AI with Cloud Shell.

### Why This Approach?
✓ **No local setup** - Everything browser-based  
✓ **Version control** - Track all code changes in GitHub  
✓ **Professional workflow** - Industry standard  
✓ **Easy collaboration** - Share with team members  
✓ **Reproducible** - Anyone can clone and train  
✓ **Pre-authenticated** - Cloud Shell knows your GCP project  

---

## STEP 1: Push Your Code to GitHub

### 1.1: Create GitHub Repository

1. Go to [GitHub.com](https://github.com)
2. Click **+** icon (top-right) → **New repository**
3. Name it: `ocr-training` or similar
4. Click **Create repository**
5. Copy the repository URL (looks like: `https://github.com/YOUR_USERNAME/ocr-training.git`)

### 1.2: Push Your Code Locally

On your computer, in PowerShell:

```powershell
cd C:\Users\brosi\Desktop\SEG4180\Project

# Initialize git (if not already)
git init

# Add all files
git add .

# Commit
git commit -m "Initial OCR training setup"

# Add remote (replace with YOUR repo URL)
git remote add origin https://github.com/YOUR_USERNAME/ocr-training.git

# Push to GitHub
git branch -M main
git push -u origin main
```

**Verify:** Go to your GitHub repo URL - you should see all your files there!

### 1.3: Create GitHub Personal Access Token (for authentication)

This lets Cloud Shell authenticate with GitHub:

1. Go to https://github.com/settings/tokens
2. Click **Generate new token** → **Generate new token (classic)**
3. Name it: `cloud-shell-access`
4. Check only: **repo** (full control of private repos)
5. Click **Generate token** at bottom
6. **COPY THE TOKEN** - you'll use it once
7. **Save it somewhere secure** - you won't see it again

---

## STEP 2: Open Cloud Shell

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Click **>_** icon (top-right) to open Cloud Shell
3. Wait for terminal to appear at bottom

---

## STEP 3: Automated Setup (Copy & Paste)

Copy this entire script and paste into Cloud Shell:

```bash
#!/bin/bash

# ===== CONFIGURATION =====
export PROJECT_ID="your-project-id"              # CHANGE THIS
export GITHUB_USERNAME="your-github-username"    # CHANGE THIS
export GITHUB_REPO="ocr-training"                # Your repo name
export GITHUB_TOKEN="ghp_xxxxxxxxxxxx"           # CHANGE THIS (Your token from Step 1.3)
export REGION="us-central1"
export BUCKET_NAME="ocr-data-$(shuf -i 10000-99999 -n 1)"
export IMAGE_NAME="ocr-classifier-training"

echo "=========================================="
echo "Vertex AI + GitHub - Cloud Shell Setup"
echo "=========================================="
echo "Project:       $PROJECT_ID"
echo "GitHub Repo:   $GITHUB_USERNAME/$GITHUB_REPO"
echo "Region:        $REGION"
echo "Bucket:        $BUCKET_NAME"
echo "=========================================="
echo ""

# Step 1: Set project
echo "[1/7] Setting GCP project..."
gcloud config set project $PROJECT_ID

# Step 2: Enable services
echo "[2/7] Enabling services..."
gcloud services enable aiplatform.googleapis.com
gcloud services enable artifactregistry.googleapis.com
gcloud services enable cloudbuild.googleapis.com
gcloud services enable storage-api.googleapis.com

# Step 3: Create bucket
echo "[3/7] Creating GCS bucket..."
gsutil mb -p $PROJECT_ID -l $REGION gs://$BUCKET_NAME

# Step 4: Clone from GitHub
echo "[4/7] Cloning code from GitHub..."
cd ~
git clone https://$GITHUB_USERNAME:$GITHUB_TOKEN@github.com/$GITHUB_USERNAME/$GITHUB_REPO.git
cd $GITHUB_REPO

# Step 5: Create Artifact Registry
echo "[5/7] Creating Artifact Registry..."
gcloud artifacts repositories create vertex-ai \
    --repository-format=docker \
    --location=$REGION 2>/dev/null || echo "  (Already exists)"

# Step 6: Build Docker image
echo "[6/7] Building Docker image from GitHub code..."
echo "  (This takes 10-15 minutes...)"
IMAGE_URI="$REGION-docker.pkg.dev/$PROJECT_ID/vertex-ai/$IMAGE_NAME:latest"

gcloud builds submit \
    --tag=$IMAGE_URI \
    --dockerfile=Dockerfile.training \
    --region=$REGION

if [ $? -ne 0 ]; then
    echo "✗ Docker build failed"
    exit 1
fi

# Step 7: Upload data to GCS
echo "[7/7] Uploading training data to GCS..."
gsutil -m cp -r data/difficulty_labels gs://$BUCKET_NAME/data/ 2>/dev/null || true
gsutil -m cp -r data/processed gs://$BUCKET_NAME/data/ 2>/dev/null || true

echo ""
echo "=========================================="
echo "✓ Setup Complete!"
echo "=========================================="
echo ""
echo "Configuration Summary:"
echo "  Project ID:    $PROJECT_ID"
echo "  Bucket:        gs://$BUCKET_NAME"
echo "  Docker Image:  $IMAGE_URI"
echo ""
echo "Next: Submit your training job:"
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

**Fill in these values before pasting:**
- `your-project-id` → Your GCP Project ID
- `your-github-username` → Your GitHub username
- `ghp_xxxxxxxxxxxx` → Your token from Step 1.3

---

## STEP 4: Submit Training Job

In Cloud Shell, after the script completes:

```bash
# Install Python libraries
pip install google-cloud-aiplatform google-cloud-storage

# Set your values (from the setup script output)
export PROJECT_ID="your-project-id"
export BUCKET_NAME="ocr-data-12345"
export REGION="us-central1"
export IMAGE_URI="us-central1-docker.pkg.dev/your-project-id/vertex-ai/ocr-classifier-training:latest"

# Navigate to project directory
cd ~/ocr-training

# Submit training job
python scripts/submit_vertex_training.py \
    --project-id $PROJECT_ID \
    --region $REGION \
    --bucket-name $BUCKET_NAME \
    --image-uri $IMAGE_URI
```

**Success output:**
```
================================================================================
✓ Job submitted successfully!

Job Resource Name:
  projects/123456789/locations/us-central1/trainingPipelines/1234567890
================================================================================
```

---

## STEP 5: Monitor Training

In Cloud Shell:

```bash
# Stream live training logs
gcloud ai custom-jobs stream-logs TRAINING_PIPELINE_ID --region=us-central1

# List all jobs
gcloud ai custom-jobs list --region=us-central1
```

Or in web UI:
- Go to: https://console.cloud.google.com/vertex-ai/training/custom-jobs

---

## Workflow for Future Training Runs

Once set up, training again is easy:

**1. Make code changes locally and push to GitHub:**
```powershell
git add .
git commit -m "Improved model architecture"
git push origin main
```

**2. In Cloud Shell, pull latest and re-train:**
```bash
cd ~/ocr-training
git pull origin main

# Update Docker image
gcloud builds submit --tag=$IMAGE_URI --dockerfile=Dockerfile.training --region=$REGION

# Re-submit training
python scripts/submit_vertex_training.py --project-id $PROJECT_ID ...
```

---

## PART 1: GCP Account & Project Setup

### Step 1.1: Create/Select a GCP Project

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Sign in with your Google account
3. At the top, click the **Project dropdown** (where it says "Select a Project")
4. Click **NEW PROJECT**
5. Enter project name: `ocr-training` or similar
6. Click **CREATE**
7. Wait 1-2 minutes for the project to be created
8. Once ready, click back on the project dropdown and select your new project

**After selection**, you should see the project name/ID in the top bar.

### Step 1.2: Find Your Project ID

You'll need your Project ID for later commands.

1. In [Cloud Console](https://console.cloud.google.com/), click the **Settings icon** (⚙️) in top-right
2. Go to **Project Settings**
3. Copy the **Project ID** (3rd field) - it looks like: `ocr-training-12345`
4. **Save this somewhere** - you'll use it in commands below

---

## PART 2: Install & Configure Google Cloud CLI

### Step 2.1: Install Google Cloud SDK

**Windows:**

1. Download installer from: https://cloud.google.com/sdk/docs/install#windows
2. Run the `.exe` file and follow prompts (default settings are fine)
3. Select "Yes" when asked to run Google Cloud initialization
4. Open **PowerShell as Administrator** and run:
   ```powershell
   gcloud init
   ```

**macOS/Linux:**

```bash
curl https://sdk.cloud.google.com | bash
exec -l $SHELL
gcloud init
```

### Step 2.2: Authenticate with GCP

During `gcloud init`, you'll be asked to log in. Follow the browser prompt to authorize your Google account.

After login, you'll be asked to select a project - **choose the project you created** (e.g., `ocr-training`).

### Step 2.3: Verify Installation

Open **PowerShell** (or Terminal on Mac/Linux) and run:

```powershell
gcloud --version
gcloud config list
```

You should see:

- Google Cloud SDK version
- Your default project listed

---

## PART 3: Enable Required Google Cloud Services

Think of GCP services like apps - you need to enable them before using them.

### Step 3.1: Enable Services

Run these commands in PowerShell:

```powershell
# Set your project ID (replace with YOUR actual project ID)
$PROJECT_ID = "ocr-training-12345"

gcloud config set project $PROJECT_ID

# Enable required services (this takes 1-2 minutes)
gcloud services enable aiplatform.googleapis.com
gcloud services enable artifactregistry.googleapis.com
gcloud services enable cloudbuild.googleapis.com
gcloud services enable storage-api.googleapis.com
gcloud services enable cloudresourcemanager.googleapis.com
```

**What these do:**

- `aiplatform` = Vertex AI training
- `artifactregistry` = Docker image storage
- `cloudbuild` = Builds Docker images
- `storage-api` = Google Cloud Storage (GCS)
- `cloudresourcemanager` = Manages resources

---

## PART 4: Create Google Cloud Storage (GCS) Bucket

GCS is like Google Drive for data. You'll store your training data here.

### Step 4.1: Create a Bucket

Run in PowerShell:

```powershell
# Set variables
$PROJECT_ID = "ocr-training-12345"
$BUCKET_NAME = "ocr-data-$(Get-Random -Minimum 10000 -Maximum 99999)"
$REGION = "us-central1"

# Create bucket
gsutil mb -p $PROJECT_ID -l $REGION gs://$BUCKET_NAME

# Verify
gsutil ls
```

You should see your bucket listed, like:

```
gs://ocr-data-45678
```

**Save the bucket name** - you'll use it below.

### Step 4.2: Create Folder Structure in Bucket

```powershell
# Create empty "folders" in the bucket
echo $null | gsutil cp - gs://$BUCKET_NAME/data/.keep
echo $null | gsutil cp - gs://$BUCKET_NAME/models/.keep
echo $null | gsutil cp - gs://$BUCKET_NAME/logs/.keep
```

---

## PART 5: Upload Your Data to GCS (Using Local gcloud CLI)

**FASTEST METHOD:** Use `gsutil` from your local PowerShell terminal (much faster than web upload)

### Step 5.1: Verify gcloud CLI is Installed

In PowerShell:

```powershell
gcloud --version
gsutil --version
```

Both should show version numbers. If not, install from: https://cloud.google.com/sdk/docs/install

### Step 5.2: Authenticate gcloud

```powershell
# Login to GCP
gcloud auth login

# Set your project
$PROJECT_ID = "ocr-training-12345"  # Replace with your actual Project ID
gcloud config set project $PROJECT_ID

# Verify
gcloud config list
```

You should see your project listed.

### Step 5.3: Create GCS Bucket

```powershell
# Set variables
$PROJECT_ID = "ocr-training-12345"  # Your Project ID
$BUCKET_NAME = "ocr-data-$(Get-Random -Minimum 10000 -Maximum 99999)"
$REGION = "us-central1"

# Create bucket
gsutil mb -p $PROJECT_ID -l $REGION gs://$BUCKET_NAME

# Verify
gsutil ls
```

You should see:
```
gs://ocr-data-12345
```

**Save your bucket name** - you'll need it for the training script.

### Step 5.4: Upload Data to GCS (FAST METHOD)

**IMPORTANT:** This is MUCH faster than web upload. The `-m` flag uses parallel uploads.

Run in PowerShell from your project directory:

```powershell
# Navigate to your project
cd C:\Users\brosi\Desktop\SEG4180\Project

# Set your bucket name
$BUCKET_NAME = "ocr-data-12345"  # Replace with your actual bucket name

# Upload classifier training data
Write-Host "Uploading classifier data..." -ForegroundColor Cyan
gsutil -m cp -r data\difficulty_labels gs://$BUCKET_NAME/data/

# Upload processed OCR data (train/val/test CSVs)
Write-Host "Uploading processed data..." -ForegroundColor Cyan
gsutil -m cp -r data\processed gs://$BUCKET_NAME/data/

# Verify upload completed
Write-Host "Verifying upload..." -ForegroundColor Cyan
gsutil ls -r gs://$BUCKET_NAME/data/ | head -20
```

**What's happening:**
- `gsutil -m` = Multi-threaded upload (uses multiple connections = FAST)
- `-r` = Recursive (upload entire folders)
- `head -20` = Show first 20 files to verify

**Expected output:**
```
gs://ocr-data-12345/data/difficulty_labels/easy/image1.jpg
gs://ocr-data-12345/data/difficulty_labels/easy/image2.jpg
gs://ocr-data-12345/data/difficulty_labels/medium/...
gs://ocr-data-12345/data/difficulty_labels/hard/...
gs://ocr-data-12345/data/processed/train.csv
gs://ocr-data-12345/data/processed/val.csv
gs://ocr-data-12345/data/processed/test.csv
```

### Speed Comparison

| Method | Speed | Best For |
|--------|-------|----------|
| Cloud Shell upload button | Slow (1-2 MB/s) | Small files only |
| `gsutil -m` (local terminal) | **Fast (50-100 MB/s)** | **Large datasets ✓** |
| Manual web drag-drop | Very slow | Not recommended |

**For large datasets like yours, `gsutil -m` is 50x faster!** ⚡

### Verify Everything Uploaded

```powershell
# Check bucket size
gsutil du -sh gs://$BUCKET_NAME

# List all files
gsutil ls -r gs://$BUCKET_NAME/data/

# Check specific folder
gsutil ls -h gs://$BUCKET_NAME/data/difficulty_labels/easy/ | wc -l
```

Done! Your data is on Google Cloud Storage and ready for training.

---

## PART 6: Set Up Docker Image (Container)

Docker packages your code + dependencies into a self-contained "container" that runs on GCP.

### Step 6.1: Create Training Dockerfile

In your project root, create a file named `Dockerfile.training`:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Install GCP libraries
RUN pip install --no-cache-dir \
    google-cloud-storage==2.14.0 \
    google-cloud-aiplatform==1.50.0

# Copy source code
COPY . .

# Default command (can be overridden)
ENTRYPOINT ["python", "-m"]
CMD ["src.classifier.train_vertex"]
```

**What this does:**

1. Starts with Python 3.10 image
2. Installs system dependencies (Tesseract OCR, etc.)
3. Installs Python packages (TensorFlow, etc.)
4. Installs Google Cloud libraries
5. Copies your code into the container

### Step 6.2: Configure Docker

**Windows:**

1. Download Docker Desktop from: https://www.docker.com/products/docker-desktop
2. Install and restart your computer
3. Open PowerShell and verify:
   ```powershell
   docker --version
   ```

**macOS/Linux:** Similar process, or use package manager

---

## PART 7: Create Training Script for GCS

Your current training script (`src/classifier/train.py`) reads local files. We need a version that downloads from GCS.

### Step 7.1: Create Training Script

Create file: `src/classifier/train_vertex.py`

```python
import os
import argparse
import json
import tensorflow as tf
from google.cloud import storage
from pathlib import Path

from src.classifier.model import build_difficulty_classifier
from src.classifier.dataset import load_difficulty_dataset

# === HYPERPARAMETERS ===
BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 1e-3
IMAGE_SIZE = (128, 128)
MODEL_SAVE_DIR = "models/classifier"


def download_folder_from_gcs(bucket_name, gcs_prefix, local_dir):
    """Download folder from GCS to local disk."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    os.makedirs(local_dir, exist_ok=True)

    blobs = bucket.list_blobs(prefix=gcs_prefix)
    for blob in blobs:
        if blob.name.endswith('/'):
            continue

        # Create local path
        relative_path = blob.name.replace(gcs_prefix, '').lstrip('/')
        local_file = os.path.join(local_dir, relative_path)
        os.makedirs(os.path.dirname(local_file), exist_ok=True)

        # Download
        blob.download_to_filename(local_file)
        print(f"Downloaded: {local_file}")

    print(f"✓ Downloaded data to {local_dir}")


def upload_folder_to_gcs(bucket_name, local_dir, gcs_prefix):
    """Upload folder to GCS."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    for root, dirs, files in os.walk(local_dir):
        for file in files:
            local_file = os.path.join(root, file)
            relative_path = os.path.relpath(local_file, local_dir)
            blob_path = f"{gcs_prefix}/{relative_path}".replace("\\", "/")

            blob = bucket.blob(blob_path)
            blob.upload_from_filename(local_file)
            print(f"Uploaded: gs://{bucket_name}/{blob_path}")

    print(f"✓ Uploaded model to gs://{bucket_name}/{gcs_prefix}")


def train_classifier_vertex(
    gcs_bucket: str,
    gcs_data_prefix: str = "data/difficulty_labels",
    gcs_model_prefix: str = "models/classifier",
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
):
    """Train classifier, downloading data from GCS and uploading results."""

    print("=" * 60)
    print("VERTEX AI CLASSIFIER TRAINING")
    print("=" * 60)

    # Step 1: Download data from GCS
    print("\n[1/5] Downloading data from GCS...")
    local_data_dir = "/tmp/data"
    download_folder_from_gcs(gcs_bucket, gcs_data_prefix, local_data_dir)

    # Step 2: Load dataset
    print("\n[2/5] Loading dataset...")
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs("logs/tensorboard/classifier", exist_ok=True)

    train_ds, val_ds, test_ds = load_difficulty_dataset(
        data_dir=local_data_dir,
        image_size=IMAGE_SIZE,
        batch_size=batch_size,
    )
    print(f"  Train batches: {len(train_ds)}")
    print(f"  Val batches: {len(val_ds)}")
    print(f"  Test batches: {len(test_ds)}")

    # Step 3: Build and compile model
    print("\n[3/5] Building model...")
    model = build_difficulty_classifier(
        input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 1),
        num_classes=3,
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    # Step 4: Setup callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(MODEL_SAVE_DIR, "best_model.keras"),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir="logs/tensorboard/classifier",
            histogram_freq=1,
        ),
    ]

    # Step 5: Train
    print("\n[4/5] Starting training...")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )

    # Evaluate
    print("\n[5/5] Evaluating on test set...")
    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    print(f"  Test Loss: {test_loss:.4f}")
    print(f"  Test Accuracy: {test_acc:.4f}")

    # Save final model
    model.save(os.path.join(MODEL_SAVE_DIR, "final_model.keras"))

    # Upload results back to GCS
    print("\nUploading results to GCS...")
    upload_folder_to_gcs(gcs_bucket, MODEL_SAVE_DIR, gcs_model_prefix)
    upload_folder_to_gcs(gcs_bucket, "logs/tensorboard/classifier", f"{gcs_model_prefix}/logs")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"\nResults stored in: gs://{gcs_bucket}/{gcs_model_prefix}/")

    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train classifier on Vertex AI")
    parser.add_argument("--gcs-bucket", required=True, help="GCS bucket name (e.g., ocr-data-12345)")
    parser.add_argument("--gcs-data-prefix", default="data/difficulty_labels", help="Path to data in GCS")
    parser.add_argument("--gcs-model-prefix", default="models/classifier", help="Path to save models in GCS")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE, help="Learning rate")

    args = parser.parse_args()

    train_classifier_vertex(
        gcs_bucket=args.gcs_bucket,
        gcs_data_prefix=args.gcs_data_prefix,
        gcs_model_prefix=args.gcs_model_prefix,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
```

---

## PART 8: Build and Push Docker Image

Now we create a Docker image from your code and push it to Google Cloud.

### Step 8.1: Set Variables

Open PowerShell and set these:

```powershell
$PROJECT_ID = "ocr-training-12345"  # Your project ID
$REGION = "us-central1"
$IMAGE_NAME = "ocr-classifier-training"
$IMAGE_TAG = "latest"
```

### Step 8.2: Create Artifact Registry Repository

This is where Docker images are stored in Google Cloud.

```powershell
gcloud artifacts repositories create vertex-ai `
    --repository-format=docker `
    --location=$REGION `
    --project=$PROJECT_ID
```

If successful, you should see:

```
Created repository [vertex-ai].
```

### Step 8.3: Build Image Locally (Optional - for testing)

Test that your Docker image builds:

```powershell
cd C:\Users\brosi\Desktop\SEG4180\Project

docker build -f Dockerfile.training -t $IMAGE_NAME`:$IMAGE_TAG .
```

⏱️ **This takes 5-10 minutes** (first time). You'll see lots of text as packages install.

### Step 8.4: Push Image to Google Cloud

**Method 1: Using Cloud Build (Recommended for beginners)**

Cloud Build (Google's CI/CD) builds the image for you:

```powershell
$IMAGE_URI = "$REGION-docker.pkg.dev/$PROJECT_ID/vertex-ai/$IMAGE_NAME`:$IMAGE_TAG"

gcloud builds submit `
    --tag=$IMAGE_URI `
    --dockerfile=Dockerfile.training `
    --region=$REGION
```

This will:

1. Upload your code to Google Cloud
2. Build the Docker image there
3. Store it in Artifact Registry

**Progress indicators:**

- `BUILD QUEUED` → Waiting
- `BUILD STARTED` → Running
- `BUILD SUCCESS` → Done! ✓

⏱️ **This takes 10-15 minutes** the first time.

Once complete, you should see:

```
Image [us-central1-docker.pkg.dev/ocr-training-12345/vertex-ai/ocr-classifier-training:latest]
successfully built and pushed.
```

**Save the full image URI** - you'll use it in the next step.

---

## PART 9: Submit Training Job to Vertex AI

Now the fun part - submitting your training job!

### Step 9.1: Create Submission Script

Create file: `scripts/submit_vertex_training.py`

```python
import argparse
from google.cloud import aiplatform


def submit_training_job(
    project_id: str,
    region: str,
    bucket_name: str,
    image_uri: str,
    job_name: str = "classifier-training",
    epochs: int = 30,
    batch_size: int = 64,
    machine_type: str = "n1-standard-4",
    gpu_type: str = None,
    gpu_count: int = 0,
):
    """
    Submit a training job to Vertex AI.

    Args:
        project_id: GCP project ID
        region: GCP region (e.g., us-central1)
        bucket_name: GCS bucket name
        image_uri: Docker image URI
        job_name: Display name for the job
        epochs: Number of training epochs
        batch_size: Batch size
        machine_type: Machine type (n1-standard-4, n1-standard-8, etc.)
        gpu_type: GPU type (NVIDIA_TESLA_K80, NVIDIA_TESLA_T4, etc.)
        gpu_count: Number of GPUs
    """

    # Initialize Vertex AI
    aiplatform.init(project=project_id, location=region)

    # Create contrainer job
    job = aiplatform.CustomContainerTrainingJob(
        display_name=job_name,
        container_uri=image_uri,
    )

    # Prepare command-line arguments
    args = [
        "--gcs-bucket", bucket_name,
        "--gcs-data-prefix", "data/difficulty_labels",
        "--gcs-model-prefix", "models/classifier",
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
    ]

    print("=" * 70)
    print("SUBMITTING VERTICX AI TRAINING JOB")
    print("=" * 70)
    print(f"Project ID:      {project_id}")
    print(f"Region:          {region}")
    print(f"Job Name:        {job_name}")
    print(f"Image:           {image_uri}")
    print(f"Machine Type:    {machine_type}")
    if gpu_type:
        print(f"GPU:             {gpu_count}x {gpu_type}")
    print(f"Epochs:          {epochs}")
    print(f"Batch Size:      {batch_size}")
    print(f"Data:            gs://{bucket_name}/data/difficulty_labels")
    print("=" * 70)

    # Submit job
    model = job.run(
        args=args,
        replica_count=1,
        machine_type=machine_type,
        accelerator_type=gpu_type if gpu_type else None,
        accelerator_count=gpu_count if gpu_type else 0,
        sync=False,  # Don't wait for job to complete
    )

    print(f"\n✓ Job submitted successfully!")
    print(f"\nJob Resource Name: {model.resource_name}")
    print(f"\nMonitor your training job:")
    print(f"  https://console.cloud.google.com/vertex-ai/training/custom-jobs?referrer=search&project={project_id}")

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Submit Vertex AI training job")
    parser.add_argument("--project-id", required=True, help="GCP project ID")
    parser.add_argument("--region", default="us-central1", help="GCP region")
    parser.add_argument("--bucket-name", required=True, help="GCS bucket name")
    parser.add_argument("--image-uri", required=True, help="Docker image URI")
    parser.add_argument("--job-name", default="classifier-training", help="Training job name")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--machine-type", default="n1-standard-4", help="Machine type")
    parser.add_argument("--gpu-type", default=None, help="GPU type (e.g., NVIDIA_TESLA_K80)")
    parser.add_argument("--gpu-count", type=int, default=0, help="Number of GPUs")

    args = parser.parse_args()

    submit_training_job(
        project_id=args.project_id,
        region=args.region,
        bucket_name=args.bucket_name,
        image_uri=args.image_uri,
        job_name=args.job_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        machine_type=args.machine_type,
        gpu_type=args.gpu_type,
        gpu_count=args.gpu_count,
    )
```

### Step 9.2: Submit the Job

**PowerShell:**

```powershell
# Set your values
$PROJECT_ID = "ocr-training-12345"
$REGION = "us-central1"
$BUCKET_NAME = "ocr-data-45678"
$IMAGE_URI = "us-central1-docker.pkg.dev/ocr-training-12345/vertex-ai/ocr-classifier-training:latest"

cd C:\Users\brosi\Desktop\SEG4180\Project

python scripts/submit_vertex_training.py `
    --project-id $PROJECT_ID `
    --region $REGION `
    --bucket-name $BUCKET_NAME `
    --image-uri $IMAGE_URI `
    --job-name "classifier-training-exp1" `
    --epochs 30 `
    --batch-size 64 `
    --machine-type "n1-standard-4"
```

**Expected output:**

```
======================================================================
SUBMITTING VERTEX AI TRAINING JOB
======================================================================
Project ID:      ocr-training-12345
Region:          us-central1
Job Name:        classifier-training-exp1
Image:           us-central1-docker.pkg.dev/ocr-training-12345/vertex-ai/ocr-classifier-training:latest
Machine Type:    n1-standard-4
Epochs:          30
Batch Size:      64
Data:            gs://ocr-data-45678/data/difficulty_labels
======================================================================

✓ Job submitted successfully!

Job Resource Name: projects/123456789/locations/us-central1/trainingPipelines/1234567890

Monitor your training job:
  https://console.cloud.google.com/vertex-ai/training/custom-jobs?referrer=search&project=ocr-training-12345
```

**Congratulations! Your job is now running on Google Cloud!** 🎉

---

## PART 10: Monitor Your Training Job

### Option A: Web Console (Easiest)

1. Click the link from the output above, OR
2. Go to: https://console.cloud.google.com/vertex-ai/training/custom-jobs
3. Select your project
4. You should see your training job listed
5. Click on it to see:
   - Status (Running, Completed, Failed)
   - Logs
   - Resource usage (CPU, GPU, Memory)
   - Training progress

### Option B: Command Line

```powershell
# List all training jobs
gcloud ai custom-jobs list --region=us-central1 --project=$PROJECT_ID

# Stream logs for a job
# (replace JOB_ID with ID from the output above, e.g., 1234567890)
gcloud ai custom-jobs stream-logs JOB_ID --region=us-central1 --project=$PROJECT_ID
```

### Expected Training Timeline

For a typical training run:

- **Minutes 0-5**: Job starts, pulls Docker image
- **Minutes 5-10**: Data downloads from GCS
- **Minutes 10+**: Training starts
  - Epoch progress appears
  - After each epoch: loss, accuracy, validation metrics

You'll see output like:

```
Epoch 1/30
125/125 [==============================] - 45s 362ms/step - loss: 1.0234 - accuracy: 0.6523 - val_loss: 0.8901 - val_accuracy: 0.7123
Epoch 2/30
...
```

---

## PART 11: Download Training Results

Once training completes, download your models and logs:

```powershell
# Download model files
gsutil -m cp -r gs://$BUCKET_NAME/models/classifier ./local_results/

# Download logs for TensorBoard
gsutil -m cp -r gs://$BUCKET_NAME/models/classifier/logs ./local_results/

# View locally
cd local_results
tree  # or: Get-ChildItem -Recurse (PowerShell)
```

---

## Glossary - Key Terms Explained

| Term                           | Meaning                                             |
| ------------------------------ | --------------------------------------------------- |
| **Project ID**                 | Unique identifier for your GCP project              |
| **GCS (Google Cloud Storage)** | Google's file storage (like Google Drive)           |
| **Bucket**                     | A folder in GCS for storing files                   |
| **Docker**                     | Package your code + dependencies into a container   |
| **Image**                      | Docker template (like a blueprint)                  |
| **Container**                  | Running instance of a Docker image                  |
| **Artifact Registry**          | Where Docker images are stored in Google Cloud      |
| **Vertex AI**                  | Google's managed ML platform                        |
| **Custom Training Job**        | Running your code on Google's machines              |
| **Machine Type**               | Type of computer (n1-standard-4 = 4 CPUs, 15GB RAM) |
| **GPU**                        | Graphics processor for faster training              |
| **TensorBoard**                | Tool to visualize training progress                 |

---

## Troubleshooting

### Problem: "Permission denied" or "not authorized"

**Solution:** Make sure you're logged in correctly:

```powershell
gcloud auth list
gcloud auth application-default login
```

### Problem: "Bucket not found" when uploading

**Solution:** Check bucket name:

```powershell
gsutil ls
```

### Problem: Docker build fails

**Solution:** Make sure you're in the project directory:

```powershell
cd C:\Users\brosi\Desktop\SEG4180\Project
dir Dockerfile.training  # Should exist
```

### Problem: Training job stays in "QUEUED" state

**Solution:**

1. Check for errors in the job logs (Web Console)
2. Make sure your Docker image built successfully
3. Verify GCS bucket permissions

### Problem: Out of memory or GPU errors

**Solution:** Try larger machine type:

```powershell
# Instead of n1-standard-4, use:
--machine-type "n1-standard-8"  # 8 CPUs, 30GB RAM
```

---

## Cost Estimation

**Typical costs for one training run:**

| Resource                   | Cost       | Duration        |
| -------------------------- | ---------- | --------------- |
| n1-standard-4 CPU          | $0.19/hour | ~1 hour = $0.19 |
| Data transfer (1GB upload) | ~$0.12     | One-time        |
| GCS storage (1GB/month)    | $0.02      | Stored          |
| **Total for 1 run**        | ~**$0.33** | -               |

**You have $300 free credit,** so you can run ~900 training jobs before spending money.

---

## Next Steps

After successful training:

1. **Download model** from GCS
2. **Deploy to API** using Cloud Run
3. **Set up hyperparameter tuning** for better results
4. **Train OCR model** using similar steps

---

## ⚡ Quick Reference - Efficient Local Terminal Upload

**For fast uploads from your local terminal (50x faster than web):**

```powershell
# 1. Install & login
gcloud init
gcloud auth login

# 2. Set project & variables
$PROJECT_ID = "ocr-training-12345"
$BUCKET_NAME = "ocr-data-12345"
gcloud config set project $PROJECT_ID

# 3. Create bucket (one-time)
gsutil mb -p $PROJECT_ID -l us-central1 gs://$BUCKET_NAME

# 4. UPLOAD DATA (FAST - parallel transfer with -m flag)
cd C:\Users\brosi\Desktop\SEG4180\Project

# Upload everything
gsutil -m cp -r data\difficulty_labels gs://$BUCKET_NAME/data/
gsutil -m cp -r data\processed gs://$BUCKET_NAME/data/

# 5. Verify upload
gsutil ls -r gs://$BUCKET_NAME/data/ | head -20

# 6. Monitor upload progress (open new terminal)
gcloud compute operations list
```

**Why `gsutil -m` is FAST:**
- ✓ Parallel uploads (50-100 MB/s vs 1-2 MB/s web upload)
- ✓ Automatic retries on failure
- ✓ Works with intermittent connections
- ✓ Shows progress indicators

**For large datasets, this finishes in 2-5 minutes instead of 1+ hour!**

---

## Full Complete Workflow

Copy this entire script to automate everything locally:

```powershell
# Save as upload_and_train.ps1

param(
    [string]$ProjectId = "ocr-training-12345",
    [string]$BucketName = "ocr-data-$(Get-Random -Minimum 10000 -Maximum 99999)",
    [string]$Region = "us-central1"
)

Write-Host "=== Vertex AI Training Setup ===" -ForegroundColor Cyan

# Navigate to project
cd C:\Users\brosi\Desktop\SEG4180\Project

# Set project
gcloud config set project $ProjectId
Write-Host "✓ Project set to: $ProjectId" -ForegroundColor Green

# Create bucket
gsutil mb -p $ProjectId -l $Region gs://$BucketName 2>&1 | Where-Object { $_ -notlike "*409*" }
Write-Host "✓ Bucket: gs://$BucketName" -ForegroundColor Green

# Upload data quickly using parallel transfer
Write-Host "`nUploading training data..." -ForegroundColor Yellow
gsutil -m cp -r data\difficulty_labels gs://$BucketName/data/
gsutil -m cp -r data\processed gs://$BucketName/data/
Write-Host "✓ Data uploaded" -ForegroundColor Green

# List uploaded files
Write-Host "`nVerifying upload..." -ForegroundColor Cyan
gsutil ls -r gs://$BucketName/data/ | Select-Object -First 10

Write-Host "`n=== Next Steps ===" -ForegroundColor Cyan
Write-Host "1. Push code to GitHub:`ncd ~ && git push origin main`n" -ForegroundColor White
Write-Host "2. Go to: https://console.cloud.google.com/`n" -ForegroundColor White
Write-Host "3. Open Cloud Shell (>_ button)`n" -ForegroundColor White
Write-Host "4. Run the GitHub + Cloud Shell setup from the guide`n" -ForegroundColor White
Write-Host "Variables to use:`nPROJECT_ID=$ProjectId`nBUCKET_NAME=$BucketName`n" -ForegroundColor Cyan
```

Run it:
```powershell
.\upload_and_train.ps1 -ProjectId "ocr-training-12345" -BucketName "ocr-data-12345"
```

---

## Quick Reference - Command Checklist

```powershell
# 1. Login (ONE TIME)
gcloud init
gcloud auth login

# 2. Set project
$PROJECT_ID = "ocr-training-12345"
gcloud config set project $PROJECT_ID

# 3. Enable services (ONE TIME)
gcloud services enable aiplatform.googleapis.com
gcloud services enable artifactregistry.googleapis.com
gcloud services enable cloudbuild.googleapis.com
gcloud services enable storage-api.googleapis.com

# 4. Create bucket (ONE TIME)
$BUCKET_NAME = "ocr-data-$(Get-Random -Minimum 10000 -Maximum 99999)"
gsutil mb gs://$BUCKET_NAME

# 5. UPLOAD DATA (FAST via local terminal)
cd C:\Users\brosi\Desktop\SEG4180\Project
gsutil -m cp -r data\difficulty_labels gs://$BUCKET_NAME/data/
gsutil -m cp -r data\processed gs://$BUCKET_NAME/data/

# 6. Verify
gsutil ls -r gs://$BUCKET_NAME/data/

# 7. Push code to GitHub
git push origin main

# 8. In Cloud Shell: Run GitHub setup script (from Step 3 of this guide)

# 9. Submit training (from Cloud Shell)
python scripts/submit_vertex_training.py --project-id $PROJECT_ID --bucket-name $BUCKET_NAME --image-uri ...

# 10. Monitor training
gcloud ai custom-jobs list --region=us-central1 --project=$PROJECT_ID

# 11. Download results
gsutil -m cp -r gs://$BUCKET_NAME/models/classifier ./results/
```

---

## Recommended Workflow

| Step | Where | How |
|------|-------|-----|
| **Setup** | Local Terminal | 5 minutes - run gcloud commands |
| **Upload Data** | Local Terminal | 2-5 min (with `gsutil -m`) |
| **Push Code** | Local Terminal | 1 min - `git push` |
| **Build & Train** | Cloud Shell | 20+ min (Docker build + training) |
| **Monitor** | Web UI or Cloud Shell | Real-time logs |
| **Download Results** | Local Terminal | 5 min - `gsutil -m cp` |

Happy training! 🚀
