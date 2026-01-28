#!/usr/bin/env python3
"""
Experiment Registration Script
==============================

Register experiments from VIT/BlindSpot to the central experiment index.

Usage:
    python register_experiment.py \
        --experiment_id "VIT-20251201-cnn-dilated-01" \
        --project VIT \
        --topic cnn \
        --status completed \
        --entry_point "scripts/run.py" \
        --config_path "configs/cnn_dilated.yaml" \
        --output_path "lightning_logs/version_42" \
        --metrics_summary "R2=0.987, MAE=0.031"
"""

import argparse
import csv
import json
import os
from datetime import datetime
from pathlib import Path

# Constants
PHYSICS_AI_ROOT = Path(__file__).parent.parent
INDEX_CSV = PHYSICS_AI_ROOT / "experiments_index" / "index.csv"
INDEX_JSON = PHYSICS_AI_ROOT / "experiments_index" / "index.json"

# Valid values
VALID_PROJECTS = {"VIT", "BlindSpot", "Other"}
VALID_STATUSES = {"running", "completed", "failed", "aborted"}
VALID_PRIORITIES = {"P0", "P1", "P2"}

# CSV field order
CSV_FIELDS = [
    "experiment_id",
    "project", 
    "topic",
    "status",
    "start_time",
    "end_time",
    "entry_point",
    "config_path",
    "output_path",
    "log_path",
    "metrics_summary",
    "physics_ai_logg_path",
    "priority",
    "next_action",
    "notes"
]


def load_index() -> list[dict]:
    """Load existing index from CSV."""
    if not INDEX_CSV.exists():
        return []
    
    with open(INDEX_CSV, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return list(reader)


def save_index(records: list[dict]):
    """Save index to CSV and JSON."""
    # Save CSV
    with open(INDEX_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for record in records:
            writer.writerow({k: record.get(k, '') for k in CSV_FIELDS})
    
    # Save JSON (for script-friendly access)
    with open(INDEX_JSON, 'w', encoding='utf-8') as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Saved {len(records)} records to:")
    print(f"   - {INDEX_CSV}")
    print(f"   - {INDEX_JSON}")


def register_experiment(
    experiment_id: str,
    project: str,
    topic: str,
    status: str = "running",
    start_time: str = None,
    end_time: str = None,
    entry_point: str = "",
    config_path: str = "",
    output_path: str = "",
    log_path: str = "",
    metrics_summary: str = "",
    physics_ai_logg_path: str = "",
    priority: str = "P1",
    next_action: str = "",
    notes: str = "",
    update: bool = False
) -> dict:
    """Register a new experiment or update an existing one."""
    
    # Validate inputs
    if project not in VALID_PROJECTS:
        raise ValueError(f"Invalid project: {project}. Must be one of {VALID_PROJECTS}")
    if status not in VALID_STATUSES:
        raise ValueError(f"Invalid status: {status}. Must be one of {VALID_STATUSES}")
    if priority and priority not in VALID_PRIORITIES:
        raise ValueError(f"Invalid priority: {priority}. Must be one of {VALID_PRIORITIES}")
    
    # Load existing index
    records = load_index()
    
    # Check if experiment already exists
    existing_idx = None
    for i, r in enumerate(records):
        if r.get("experiment_id") == experiment_id:
            existing_idx = i
            break
    
    # Prepare record
    now = datetime.now().isoformat()
    record = {
        "experiment_id": experiment_id,
        "project": project,
        "topic": topic,
        "status": status,
        "start_time": start_time or now,
        "end_time": end_time or ("" if status == "running" else now),
        "entry_point": entry_point,
        "config_path": config_path,
        "output_path": output_path,
        "log_path": log_path,
        "metrics_summary": metrics_summary,
        "physics_ai_logg_path": physics_ai_logg_path,
        "priority": priority,
        "next_action": next_action,
        "notes": notes
    }
    
    if existing_idx is not None:
        if update:
            # Merge with existing record (keep non-empty values from existing)
            existing = records[existing_idx]
            for key in CSV_FIELDS:
                if not record.get(key) and existing.get(key):
                    record[key] = existing[key]
            records[existing_idx] = record
            print(f"📝 Updated experiment: {experiment_id}")
        else:
            print(f"⚠️  Experiment {experiment_id} already exists. Use --update to modify.")
            return records[existing_idx]
    else:
        records.append(record)
        print(f"✨ Registered new experiment: {experiment_id}")
    
    # Save
    save_index(records)
    
    return record


def main():
    parser = argparse.ArgumentParser(
        description="Register experiments to the central index",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Register a new completed experiment
  python register_experiment.py \\
    --experiment_id "VIT-20251201-cnn-dilated-01" \\
    --project VIT \\
    --topic cnn \\
    --status completed \\
    --metrics_summary "R2=0.987, MAE=0.031"

  # Update an existing experiment
  python register_experiment.py \\
    --experiment_id "VIT-20251201-cnn-dilated-01" \\
    --status completed \\
    --metrics_summary "R2=0.992, MAE=0.028" \\
    --update
        """
    )
    
    parser.add_argument("--experiment_id", "-e", required=True, help="Unique experiment identifier")
    parser.add_argument("--project", "-p", required=True, choices=list(VALID_PROJECTS), help="Source project")
    parser.add_argument("--topic", "-t", required=True, help="Topic category (cnn, swin, noise, etc.)")
    parser.add_argument("--status", "-s", default="running", choices=list(VALID_STATUSES), help="Experiment status")
    parser.add_argument("--start_time", help="Start time (ISO format)")
    parser.add_argument("--end_time", help="End time (ISO format)")
    parser.add_argument("--entry_point", help="Entry script path")
    parser.add_argument("--config_path", help="Config file path")
    parser.add_argument("--output_path", help="Output directory path")
    parser.add_argument("--log_path", help="Log file path")
    parser.add_argument("--metrics_summary", "-m", help="Key metrics summary string")
    parser.add_argument("--physics_ai_logg_path", help="Corresponding logg/*.md path")
    parser.add_argument("--priority", default="P1", choices=list(VALID_PRIORITIES), help="Priority level")
    parser.add_argument("--next_action", help="Next action description")
    parser.add_argument("--notes", help="Additional notes")
    parser.add_argument("--update", "-u", action="store_true", help="Update existing experiment")
    
    args = parser.parse_args()
    
    register_experiment(
        experiment_id=args.experiment_id,
        project=args.project,
        topic=args.topic,
        status=args.status,
        start_time=args.start_time,
        end_time=args.end_time,
        entry_point=args.entry_point,
        config_path=args.config_path,
        output_path=args.output_path,
        log_path=args.log_path,
        metrics_summary=args.metrics_summary,
        physics_ai_logg_path=args.physics_ai_logg_path,
        priority=args.priority,
        next_action=args.next_action,
        notes=args.notes,
        update=args.update
    )


if __name__ == "__main__":
    main()

