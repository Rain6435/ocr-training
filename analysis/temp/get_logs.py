#!/usr/bin/env python3
"""Get training job stderr logs."""

from google.cloud import logging as cloud_logging
import json

client = cloud_logging.Client(project='ocr-training-491603')

# Query for logs from the failed job
query = '''
resource.type="ml_job"
resource.labels.job_id="6312705668524539904"
severity in (ERROR, WARNING, DEFAULT)
textPayload != ""
'''

print("=" * 80)
print("TRAINING JOB LOGS (Last 50 entries)")
print("=" * 80)
print()

try:
    entries = list(client.list_entries(filter_=query, max_results=100, page_size=100))
    
    if not entries:
        print("No logs found. Trying alternative query...")
        entries = list(client.list_entries(
            filter_='resource.type="ml_job"',
            max_results=50
        ))
    
    print(f"Found {len(entries)} log entries\n")
    
    # Sort by timestamp
    entries = sorted(entries, key=lambda e: e.timestamp or '')
    
    for entry in entries[-50:]:  # Last 50
        ts = entry.timestamp.strftime('%Y-%m-%d %H:%M:%S') if entry.timestamp else 'N/A'
        severity = entry.severity or 'INFO'
        
        print(f"[{ts}] {severity}")
        
        if hasattr(entry, 'payload') and entry.payload:
            if isinstance(entry.payload, str):
                # Truncate very long lines
                lines = entry.payload.split('\n')
                for line in lines[:5]:
                    if line:
                        print(f"  {line[:120]}")
                if len(lines) > 5:
                    print(f"  ... ({len(lines)-5} more lines)")
            elif isinstance(entry.payload, dict):
                print(f"  {json.dumps(entry.payload)[:200]}")
        print()
        
except Exception as e:
    print(f"Error fetching logs: {e}")
    import traceback
    traceback.print_exc()
    print()
    print("Try accessing logs manually at:")
    print("https://console.cloud.google.com/logs/viewer?project=536990699426")
