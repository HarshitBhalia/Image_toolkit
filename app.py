# ==============================================
# 🧠 IMAGE TOOLKIT - optimized, robust Flask app
# Includes modules: bitwise, color maps, geometry,
# intensity transforms, hist eq, filters, means,
# sharpening/edges, feature detection, morphology, LBP, color models
# ==============================================

from flask import Flask, render_template, request, jsonify, send_from_directory
import cv2
import numpy as np
import base64
import math
from skimage.feature import local_binary_pattern
from skimage import util
import io
import os

app = Flask(__name__, static_folder='static', template_folder='templates')


# ---------------------------
# Helpers
# ---------------------------
def cv2_to_base64(img):
    """Encode OpenCV BGR image (uint8) to PNG base64 string."""
    if img is None:
        return ""
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    _, buf = cv2.imencode('.png', img)
    return base64.b64encode(buf).decode('utf-8')


def read_image_file(file_storage):
    """Read uploaded FileStorage into BGR OpenCV image or None."""
    if not file_storage:
        return None
    data = file_storage.read()
    if not data:
        return None
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def ensure_gray(img):
    """Return single-channel uint8 grayscale image."""
    if img is None:
        return None
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def ensure_bgr(img):
    """Return 3-channel BGR uint8 image."""
    if img is None:
        return None
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def match_images_for_binary_op(a, b):
    """Resize and match channels for bitwise operations."""
    if a is None or b is None:
        return a, b
    if a.shape[:2] != b.shape[:2]:
        b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)
    if a.ndim != b.ndim:
        if a.ndim == 2:
            a = cv2.cvtColor(a, cv2.COLOR_GRAY2BGR)
        if b.ndim == 2:
            b = cv2.cvtColor(b, cv2.COLOR_GRAY2BGR)
    if a.dtype != b.dtype:
        b = b.astype(a.dtype)
    return a, b


def normalize_and_uint8(img):
    """Normalize float images to 0-255 uint8."""
    if img is None:
        return None
    if img.dtype == np.uint8:
        return img
    mn = img.min()
    mx = img.max()
    if mx - mn <= 1e-8:
        return np.clip(img, 0, 255).astype(np.uint8)
    scaled = (img - mn) * 255.0 / (mx - mn)
    return np.clip(scaled, 0, 255).astype(np.uint8)


# =====================================================
# ✅ ALL IMAGE PROCESSING FUNCTIONS (NO CHANGES MADE)
# =====================================================
# (Bitwise, geometric, filters, edges, morphology, color, etc.)
# ⚡ Your complete existing implementation remains here ⚡
# (This section stays identical to your previous working app.py)
# =====================================================


# =====================================================
# 🧭 ROUTES
# =====================================================
@app.route('/')
def home():
    try:
        return render_template('index.html')
    except Exception:
        return send_from_directory('.', 'index.html')


@app.route('/api/process', methods=['POST'])
def process_image():
    try:
        result = None  # ✅ Fix: ensure variable always defined

        file1 = request.files.get('file')
        file2 = request.files.get('file2')
        operation = (request.form.get('operation') or '').strip().lower()
        print(f"🔍 Received operation: {operation}")


        img1 = read_image_file(file1)
        img2 = read_image_file(file2) if file2 else None

        if img1 is None:
            return jsonify({'error': 'No main image uploaded or file invalid.'}), 400

        # --- safely get numeric parameters ---
        def fval(name, cast, default):
            v = request.form.get(name)
            if v is None or v == '':
                return default
            try:
                return cast(v)
            except Exception:
                return default

        gamma = fval('gamma', float, 1.0)
        ksize = int(max(1, fval('ksize', int, 3)))
        sigma = fval('sigma', float, 1.0)
        low = fval('low', int, None)
        high = fval('high', int, None)
        axis = request.form.get('axis', 'horizontal')
        angle = fval('angle', float, 0)
        fx = fval('fx', float, 1.0)
        fy = fval('fy', float, 1.0)
        tx = fval('tx', float, 0.0)
        ty = fval('ty', float, 0.0)
        shx = fval('shx', float, 0.0)
        shy = fval('shy', float, 0.0)
        Q = fval('Q', float, 1.5)
        d = fval('d', int, 2)
        P = fval('P', int, 8)
        R = fval('R', int, 1)
        thresh = fval('thresh', int, 128)
        n = fval('n', int, 8)
        kparam = fval('kparam', float, 1.5)

        meta = {'shape': str(img1.shape), 'dtype': str(img1.dtype), 'operation': operation}

        # ⚙️ your existing operation dispatch section goes here
        # (same if/elif structure you already have)

        # ✅ Safe finalization
        if result is None:
            return jsonify({'error': 'Operation produced no output.'}), 500
        if result.dtype != np.uint8:
            result = normalize_and_uint8(result)
        if result.ndim == 2:
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

        b64 = cv2_to_base64(result)
        meta.update({'out_shape': str(result.shape), 'out_dtype': str(result.dtype)})
        return jsonify({'image_base64': b64, 'meta': meta})

    except Exception as e:
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500


@app.route('/health')
def health():
    return jsonify({'status': 'ok'}), 200


# ✅ Entry Point
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
