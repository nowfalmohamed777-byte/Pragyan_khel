from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import os
import uuid

import cv2

import detect_image_artifacts as dia
import mai as vtd

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, 'static', 'uploads')
OUT_DIR = os.path.join(BASE_DIR, 'static', 'outputs')
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

ALLOWED_IMAGE = {'.png', '.jpg', '.jpeg', '.bmp'}
ALLOWED_VIDEO = {'.mp4', '.mov', '.avi', '.mkv'}

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024  # 1GB max


def allowed_file(filename):
    _, ext = os.path.splitext(filename.lower())
    return ext in ALLOWED_IMAGE or ext in ALLOWED_VIDEO


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return redirect(url_for('index'))
    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('index'))
    if not allowed_file(file.filename):
        return 'Unsupported file type', 400

    filename = secure_filename(file.filename)
    token = uuid.uuid4().hex[:8]
    saved_name = f"{token}_{filename}"
    saved_path = os.path.join(UPLOAD_DIR, saved_name)
    file.save(saved_path)

    _, ext = os.path.splitext(filename.lower())
    if ext in ALLOWED_IMAGE:
        # run image detector
        img = dia.load_image(saved_path)
        ghost = dia.detect_ghost_regions(img)
        seam = dia.detect_seams(img)
        annotated = dia.annotate_image(img, ghost, seam)
        out_img_name = f"annotated_{saved_name}.png"
        out_img_path = os.path.join(OUT_DIR, out_img_name)
        cv2.imwrite(out_img_path, annotated)
        report_name = f"report_{saved_name}.csv"
        report_path = os.path.join(OUT_DIR, report_name)
        dia.write_report(report_path, ghost, seam)
        return render_template('result.html', type='image', annotated=os.path.join('static', 'outputs', out_img_name), report=os.path.join('static', 'outputs', report_name))

    else:
        # run video detector (may be slow)
        frames, timestamps, meta = vtd.load_video_frames(saved_path)
        intervals = vtd.compute_intervals(timestamps)
        drop_events = vtd.detect_drops(intervals)
        merge_indices = set(vtd.detect_merges(frames))
        labels = ['Normal'] * len(frames)
        notes = [''] * len(frames)
        for idx, missing in drop_events:
            if 0 <= idx < len(frames):
                labels[idx] = 'Frame Drop'
                notes[idx] = f"Estimated missing frames: {missing} (timestamp gap)"
        for idx in merge_indices:
            if 0 <= idx < len(frames) and labels[idx] == 'Normal':
                labels[idx] = 'Frame Merge'
                notes[idx] = 'Neighbor-average indicates blended frame (heuristic)'

        out_video_name = f"annotated_{saved_name}.mp4"
        out_video_path = os.path.join(OUT_DIR, out_video_name)
        out_fps = meta.get('fps') or (1000.0 / (float(intervals.mean()) if intervals.size else 25.0))
        vtd.annotate_and_write(frames, timestamps, labels, out_video_path, fps=out_fps)
        report_name = f"report_{saved_name}.csv"
        report_path = os.path.join(OUT_DIR, report_name)
        vtd.write_csv_report(report_path, timestamps, labels, notes)
        return render_template('result.html', type='video', annotated=os.path.join('static', 'outputs', out_video_name), report=os.path.join('static', 'outputs', report_name))


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
