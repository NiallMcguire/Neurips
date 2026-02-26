"""
Pre-download script for PhysionetMI dataset.
Run this ONCE on a login node before submitting training jobs.
Handles server errors with automatic retries.

Usage:
    python download_physionet.py
"""

import os
import time
import mne
mne.set_log_level("ERROR")


DATA_DIR = os.path.expanduser("~/mne_data/MNE-eegbci-data/files/eegmmidb/1.0.0")

# PhysionetMI runs needed for motor imagery:
# R04, R08, R12 = left/right hand
# R06, R10, R14 = hands/feet
RUNS = [4, 6, 8, 10, 12, 14]

# All 109 subjects
ALL_SUBJECTS = list(range(1, 110))

MAX_RETRIES = 5
RETRY_DELAY = 10  # seconds between retries


def subject_is_downloaded(subject):
    """Check if all required run files exist for a subject."""
    subj_str = f"S{subject:03d}"
    for run in RUNS:
        fname = os.path.join(DATA_DIR, subj_str, f"{subj_str}R{run:02d}.edf")
        if not os.path.exists(fname):
            return False
    return True


def download_subject(subject):
    """Download a single subject with retries."""
    from mne.datasets import eegbci

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            eegbci.load_data(subject, runs=RUNS, update_path=True, verbose=False)
            return True
        except Exception as e:
            print(f"  Attempt {attempt}/{MAX_RETRIES} failed for subject {subject}: {e}")
            if attempt < MAX_RETRIES:
                print(f"  Retrying in {RETRY_DELAY}s...")
                time.sleep(RETRY_DELAY)
            else:
                print(f"  Giving up on subject {subject} after {MAX_RETRIES} attempts.")
                return False


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    already_done = [s for s in ALL_SUBJECTS if subject_is_downloaded(s)]
    to_download = [s for s in ALL_SUBJECTS if not subject_is_downloaded(s)]

    print(f"Already downloaded: {len(already_done)}/109 subjects")
    print(f"Still need to download: {len(to_download)} subjects")

    if not to_download:
        print("All subjects already downloaded. Nothing to do.")
        return

    failed = []
    for i, subject in enumerate(to_download):
        print(f"[{i+1}/{len(to_download)}] Downloading subject {subject}...")
        success = download_subject(subject)
        if not success:
            failed.append(subject)

    print("\n--- Download Summary ---")
    print(f"Successfully downloaded: {len(to_download) - len(failed)} subjects")
    if failed:
        print(f"Failed subjects: {failed}")
        print("Re-run this script to retry failed subjects.")
    else:
        print("All subjects downloaded successfully!")


if __name__ == "__main__":
    main()
