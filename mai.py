"""Video Temporal Error Detector

Usage:
    python detect_temporal_errors.py --input input.mp4 --output annotated.mp4 --report report.csv

This script reads the video, extracts timestamps, detects frame drops (via timestamp gaps)
and frame merges (via neighbor-average MSE heuristic), then writes an annotated video and CSV.
"""
import argparse
import csv
import os
from typing import List, Tuple

import cv2
import numpy as np
import sys


def load_video_frames(video_path: str) -> Tuple[List[np.ndarray], List[float], dict]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frames = []
    timestamps = []  # milliseconds
    meta = {}

    # Try to get FPS and frame count as metadata if available
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) else -1
    meta['fps'] = fps if fps and fps > 0 else None
    meta['frame_count'] = frame_count

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # timestamp in milliseconds
        ts = cap.get(cv2.CAP_PROP_POS_MSEC)
        frames.append(frame)
        timestamps.append(ts)
        idx += 1

    cap.release()
    return frames, timestamps, meta


def compute_intervals(timestamps: List[float]) -> np.ndarray:
    ts = np.array(timestamps, dtype=float)
    intervals = np.diff(ts)
    return intervals


def detect_drops(intervals: np.ndarray, drop_factor: float = 1.5) -> List[Tuple[int, int]]:
    """Return list of (index_after_gap, missing_count) for gaps larger than drop_factor * median_interval.
    index_after_gap is the frame index where the large gap ends (i.e., the later frame index).
    """
    if len(intervals) == 0:
        return []
    median_int = float(np.median(intervals[intervals > 0]))
    if median_int <= 0:
        return []

    drop_events = []
    for i, val in enumerate(intervals):
        if val > median_int * drop_factor:
            estimated_missing = int(round(val / median_int)) - 1
            if estimated_missing < 1:
                estimated_missing = 1
            drop_events.append((i + 1, estimated_missing))
    return drop_events


def compute_mse(a: np.ndarray, b: np.ndarray) -> float:
    diff = a.astype(np.float32) - b.astype(np.float32)
    mse = float(np.mean(np.square(diff)))
    return mse


def detect_merges(frames: List[np.ndarray], merge_factor: float = 0.6) -> List[int]:
    """Heuristic: a merged/blended frame will be close to the average of its neighbors.
    Returns indices of frames classified as merges.
    """
    n = len(frames)
    if n < 3:
        return []

    # Work in grayscale for speed and robustness
    gray = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]

    mse_to_avg = []
    for i in range(1, n - 1):
        prev = gray[i - 1]
        cur = gray[i]
        nxt = gray[i + 1]
        avg = ((prev.astype(np.float32) + nxt.astype(np.float32)) / 2.0).astype(np.uint8)
        mse = compute_mse(cur, avg)
        mse_to_avg.append(mse)

    mse_arr = np.array(mse_to_avg)
    median_mse = float(np.median(mse_arr)) if mse_arr.size else 0.0

    merge_indices = []
    # low_motion_thresh avoids labeling static regions as merges
    # set a small absolute floor to avoid pathological values
    low_motion_thresh = max(5.0, median_mse * 0.1)

    for idx, mse in enumerate(mse_to_avg, start=1):
        # if current is much closer to the average of neighbors than typical,
        # consider it a merge candidate
        if median_mse > 0 and mse < median_mse * merge_factor and mse > low_motion_thresh:
            merge_indices.append(idx)

    return merge_indices


def annotate_and_write(frames: List[np.ndarray], timestamps: List[float], labels: List[str], out_path: str, fps: float = None):
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    h, w = frames[0].shape[:2]
    if fps is None or fps <= 0:
        # fallback to 25
        fps = 25.0

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    for i, frame in enumerate(frames):
        label = labels[i]
        ts = timestamps[i]
        overlay = frame.copy()
        # draw label background
        color = (0, 200, 0) if label == 'Normal' else ((0, 165, 255) if label == 'Frame Merge' else (0, 0, 255))
        cv2.rectangle(overlay, (10, 10), (360, 60), (0, 0, 0), -1)
        alpha = 0.5
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        cv2.putText(frame, f"Frame {i}", (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (230, 230, 230), 2)
        cv2.putText(frame, f"{label}", (140, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, f"{ts:.1f} ms", (260, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 1)

        writer.write(frame)

    writer.release()


def write_csv_report(out_csv: str, timestamps: List[float], labels: List[str], notes: List[str]):
    os.makedirs(os.path.dirname(out_csv) or '.', exist_ok=True)
    with open(out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['frame_index', 'timestamp_ms', 'label', 'notes'])
        for i, (ts, lab, note) in enumerate(zip(timestamps, labels, notes)):
            w.writerow([i, f"{ts:.3f}", lab, note])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', required=True, help='Input video path')
    parser.add_argument('--output', '-o', default='annotated.mp4', help='Annotated output video path')
    parser.add_argument('--report', '-r', default='report.csv', help='CSV report path')
    parser.add_argument('--drop-factor', type=float, default=1.5, help='Multiplier above median interval to flag drops')
    parser.add_argument('--merge-factor', type=float, default=0.6, help='Factor threshold for merge detection (lower==more strict)')
    args = parser.parse_args()

    # Sanitize and resolve input path (helps avoid quoting issues in PowerShell)
    input_path = args.input.strip().strip('"').strip("'")
    input_path = os.path.expanduser(input_path)
    try:
        input_path = os.path.abspath(input_path)
    except Exception:
        # Fallback to raw arg if abspath fails
        input_path = args.input

    if not os.path.exists(input_path):
        print(f"Error: input file not found: {input_path}")
        print("Make sure you provide the correct path. Example PowerShell usage:")
    print(r'  python .\detect_temporal_errors.py -i "C:\Users\nowfa\p2\20001-0053.mp4" -o "C:\Users\nowfa\p2\annotated.mp4" -r "C:\Users\nowfa\p2\report.csv"')
    # Helpful hint for common quoting mistake seen earlier
    print("Hint: don't escape quotes with backslashes in PowerShell; wrap the path in double quotes as shown above.")
    sys.exit(2)

    print(f"Loading video: {input_path}")
    frames, timestamps, meta = load_video_frames(input_path)
    n = len(frames)
    if n == 0:
        print("No frames read from video.")
        return

    print(f"Read {n} frames, meta: {meta}")

    intervals = compute_intervals(timestamps)
    median_int = float(np.median(intervals[intervals > 0])) if intervals.size else None
    if median_int:
        print(f"Median inter-frame interval: {median_int:.3f} ms (~{1000.0/median_int:.2f} fps)")

    # Detect drops
    drop_events = detect_drops(intervals, drop_factor=args.drop_factor)
    drop_map = {idx: missing for (idx, missing) in drop_events}

    # Detect merges
    merge_indices = set(detect_merges(frames, merge_factor=args.merge_factor))

    labels = ['Normal'] * n
    notes = [''] * n

    for idx, missing in drop_events:
        if 0 <= idx < n:
            labels[idx] = 'Frame Drop'
            notes[idx] = f"Estimated missing frames: {missing} (timestamp gap)"

    for idx in merge_indices:
        if 0 <= idx < n and labels[idx] == 'Normal':
            labels[idx] = 'Frame Merge'
            notes[idx] = 'Neighbor-average indicates blended frame (heuristic)'

    # Annotate and write video
    out_video = args.output
    # choose fps from meta if available
    out_fps = meta.get('fps')
    if out_fps is None or out_fps <= 0:
        # fallback to fps derived from median interval if available
        if median_int and median_int > 0:
            out_fps = 1000.0 / median_int
        else:
            out_fps = 25.0

    print(f"Writing annotated video to {out_video} (fps={out_fps:.2f})")
    annotate_and_write(frames, timestamps, labels, out_video, fps=out_fps)

    print(f"Writing CSV report to {args.report}")
    write_csv_report(args.report, timestamps, labels, notes)

    print("Done. Summary:")
    cnt_drop = sum(1 for l in labels if l == 'Frame Drop')
    cnt_merge = sum(1 for l in labels if l == 'Frame Merge')
    print(f"  Frames: {n}")
    print(f"  Frame Drops flagged: {cnt_drop}")
    print(f"  Frame Merges flagged: {cnt_merge}")


if __name__ == '__main__':
    main()
