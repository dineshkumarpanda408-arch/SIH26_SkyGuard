#!/usr/bin/env python3
"""Clean test and simulation data from the Skyguard SQLite database."""

import os
import sys

# Ensure backend directory is in python path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.join(BASE_DIR, "backend")
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from app.database import SessionLocal
from app import models


def clean_test_data(clean_simulations: bool = True, clean_anomalies: bool = False):
    db = SessionLocal()
    try:
        print("🔍 Checking test and simulation data...")
        
        # 1. Clean simulation events
        if clean_simulations:
            sim_count = db.query(models.SimulationEvent).count()
            if sim_count > 0:
                db.query(models.SimulationEvent).delete()
                print(f"  ✓ Removed {sim_count} simulation event records.")
            else:
                print("  ✓ No simulation events to clean.")

        # 2. Clean injected readings
        injected_readings = db.query(models.Reading).filter(models.Reading.is_injected == True).count()
        if injected_readings > 0:
            db.query(models.Reading).filter(models.Reading.is_injected == True).delete()
            print(f"  ✓ Removed {injected_readings} injected test readings.")
        else:
            print("  ✓ No injected test readings found.")

        # 3. Clean orphan explanations
        explanations = db.query(models.Explanation).all()
        valid_anomaly_ids = {a.id for a in db.query(models.Anomaly.id).all()}
        orphan_explanations = [e for e in explanations if e.anomaly_id not in valid_anomaly_ids]
        if orphan_explanations:
            for e in orphan_explanations:
                db.delete(e)
            print(f"  ✓ Removed {len(orphan_explanations)} orphaned explanations.")

        # 4. Optional: clean anomaly records if requested
        if clean_anomalies:
            an_count = db.query(models.Anomaly).count()
            db.query(models.Explanation).delete()
            db.query(models.Anomaly).delete()
            print(f"  ✓ Cleaned {an_count} anomaly records and associated explanations.")

        db.commit()
        print("\n✅ Test data cleanup completed successfully.")
    except Exception as e:
        db.rollback()
        print(f"\n❌ Error cleaning test data: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        db.close()


if __name__ == "__main__":
    clean_all = "--all" in sys.argv or "--anomalies" in sys.argv
    clean_test_data(clean_simulations=True, clean_anomalies=clean_all)
