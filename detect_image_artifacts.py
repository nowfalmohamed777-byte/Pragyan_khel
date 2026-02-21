"""Image artifact detector (merge-like ghosting and seam/drop-like discontinuities)

This module provides callable functions for single-image analysis used by the web app.
"""
import os
import csv
from typing import Tuple

import cv2
import numpy as np


def load_image(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img


def shift_image(img: np.ndarray, dx: int, dy: int) -> np.ndarray:
    h, w = img.shape[:2]
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    shifted = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return shifted


def compute_shift_avg_score(img_gray: np.ndarray, shifts):
    h, w = img_gray.shape
    acc = np.zeros((h, w), dtype=np.float32)
    for dx, dy in shifts:
        shifted = shift_image(img_gray, dx, dy).astype(np.float32)
        acc += shifted
    avg = acc / len(shifts)
    diff = np.abs(img_gray.astype(np.float32) - avg)
    return diff


def detect_ghost_regions(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    shifts = ((-2, 0), (2, 0), (0, -2), (0, 2))
    diff = compute_shift_avg_score(gray, shifts)
    diff_blur = cv2.GaussianBlur(diff, (7, 7), 0)
    norm = cv2.normalize(diff_blur, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, mask_low = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask_low = cv2.bitwise_not(mask_low)
    kernel = np.ones((5, 5), np.uint8)
    mask_clean = cv2.morphologyEx(mask_low, cv2.MORPH_OPEN, kernel)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel)
    return mask_clean


def detect_seams(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    abs_y = np.abs(sobel_y)
    blur = cv2.GaussianBlur(abs_y, (7, 7), 0)
    norm = cv2.normalize(blur, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, mask = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((3, 15), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=1)
    return mask


def annotate_image(img, ghost_mask, seam_mask):
    out = img.copy()
    blue = np.zeros_like(out)
    blue[:, :] = (255, 0, 0)
    out = cv2.addWeighted(out, 1.0, blue, 0.4, 0, mask=ghost_mask)
    red = np.zeros_like(out)
    red[:, :] = (0, 0, 255)
    out = cv2.addWeighted(out, 1.0, red, 0.6, 0, mask=seam_mask)
    cv2.rectangle(out, (10, 10), (360, 95), (0, 0, 0), -1)
    cv2.putText(out, 'Blue = Ghost/Merge-like region (low shift-diff)', (18, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1)
    cv2.putText(out, 'Red = High-gradient seam (possible composite boundary)', (18, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1)
    return out


def write_report(csv_path, ghost_mask, seam_mask):
    os.makedirs(os.path.dirname(csv_path) or '.', exist_ok=True)
    h, w = ghost_mask.shape
    total_pixels = h * w
    ghost_pixels = int(np.count_nonzero(ghost_mask))
    seam_pixels = int(np.count_nonzero(seam_mask))
    with open(csv_path, 'w', newline='') as f:
        wcsv = csv.writer(f)
        wcsv.writerow(['metric', 'value'])
        wcsv.writerow(['image_pixels', total_pixels])
        wcsv.writerow(['ghost_mask_pixels', ghost_pixels])
        wcsv.writerow(['seam_mask_pixels', seam_pixels])
        wcsv.writerow(['ghost_fraction', f"{ghost_pixels/total_pixels:.6f}"])
        wcsv.writerow(['seam_fraction', f"{seam_pixels/total_pixels:.6f}"])
