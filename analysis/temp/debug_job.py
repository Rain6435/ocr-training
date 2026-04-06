#!/usr/bin/env python3
"""Debug a failed Vertex AI training job."""

from google.cloud import aiplatform
import json
import sys

# Initialize
aiplatform.init(project='ocr-training-491603', location='us-central1')

# Get the specific job
job_id = '6312705668524539904'
job = aiplatform.CustomJob.get(job_id)

print("=" * 80)
print("TRAINING JOB DETAILS")
print("=" * 80)
print(f"Job Name: {job.display_name}")
print(f"State: {job.state} (5=FAILED)")
print(f"Created: {job.create_time}")
print(f"Updated: {job.update_time}")

if job.error:
    print(f"\nERROR:")
    print(f"  Code: {job.error.code}")
    print(f"  Message:\n{job.error.message}")

print("\n" + "=" * 80)
print("JOB RESOURCE NAME")
print("=" * 80)
print(job.resource_name)

# Try to check logs location
try:
    details = job.to_dict()
    print("\n" + "=" * 80)
    print("JOB CONFIGURATION")
    print("=" * 80)
    
    if 'containerSpec' in details:
        spec = details['containerSpec']
        print(f"Container Image: {spec.get('imageUri', 'N/A')}")
        print(f"Args: {spec.get('args', [])}")
        if 'env' in spec:
            for env_var in spec['env']:
                print(f"  Env: {env_var.get('name')}={env_var.get('value', '')}")
    
    if 'workerPoolSpecs' in details:
        for i, spec in enumerate(details['workerPoolSpecs']):
            print(f"\nWorker Pool {i}:")
            if 'machineSpec' in spec:
                ms = spec['machineSpec']
                print(f"  Machine: {ms.get('machineType', 'N/A')}")
                print(f"  Accelerator: {ms.get('acceleratorType', 'none')} x{ms.get('acceleratorCount', 0)}")
            print(f"  Replicas: {spec.get('replicaCount', 1)}")
except Exception as e:
    print(f"Error getting details: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("NEXT STEPS:")
print("=" * 80)
print("1. Check Cloud Logging for stderr output:")
print(f"   https://console.cloud.google.com/logs/viewer?project=536990699426")
print()
print("2. Common causes:")
print("   - Missing Python dependencies (check Docker image)")
print("   - Missing training data in GCS bucket")
print("   - Insufficient machine resources")
print("   - Authentication/permission issues")
print()
