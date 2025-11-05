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
import os
import logging
from werkzeug.middleware.proxy_fix import ProxyFix
from skimage.feature import local_binary_pattern
from skimage import util
import io

# -----------------------------------
# ✅ Basic Logging Configuration
# -----------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("image_toolkit")

# -----------------------------------
# ✅ Flask App Setup
# -----------------------------------
app = Flask(__name__, static_folder='static', template_folder='templates')

# -----------------------------------
# ✅ Production Configuration
# -----------------------------------
MAX_MB = int(os.environ.get("MAX_UPLOAD_MB", 12))  # default 12 MB
MAX_BYTES = MAX_MB * 1024 * 1024
ALLOWED_EXT = {'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'tif', 'webp'}

app.config['MAX_CONTENT_LENGTH'] = MAX_BYTES
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'change-me-for-prod')
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_port=1)

# -----------------------------------
# ✅ Helper: Validate Uploaded Files
# -----------------------------------
def allowed_filename(filename):
    if not filename:
        return False
    ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
    return ext in ALLOWED_EXT

def safe_read_file(file_storage):
    """Safely read uploaded file and check type/size."""
    if file_storage is None:
        raise ValueError("No file uploaded")
    filename = getattr(file_storage, 'filename', None)
    if not allowed_filename(filename):
        raise ValueError("Unsupported file type")
    data = file_storage.read()
    if len(data) > app.config['MAX_CONTENT_LENGTH']:
        raise ValueError(f"File too large (>{MAX_MB} MB)")
    return data

# -----------------------------------
# ✅ Core Utilities
# -----------------------------------
def cv2_to_base64(img):
    """Encode OpenCV BGR image to base64 PNG."""
    if img is None:
        return ""
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    _, buf = cv2.imencode('.png', img)
    return base64.b64encode(buf).decode('utf-8')


def read_image_file(file_storage):
    """Read uploaded image into BGR OpenCV image."""
    if not file_storage:
        return None
    data = safe_read_file(file_storage)
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def ensure_gray(img):
    if img is None:
        return None
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def ensure_bgr(img):
    if img is None:
        return None
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def match_images_for_binary_op(a, b):
    """Resize and match channels for bitwise ops."""
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
    """Normalize float image to 0-255 uint8."""
    if img is None:
        return None
    if img.dtype == np.uint8:
        return img
    mn, mx = img.min(), img.max()
    if mx - mn < 1e-8:
        return np.clip(img, 0, 255).astype(np.uint8)
    scaled = (img - mn) * 255.0 / (mx - mn)
    return np.clip(scaled, 0, 255).astype(np.uint8)


# ==================================================
# 🧮 All Operation Implementations (same as before)
# ==================================================
# (Your functions for bitwise, geometry, intensity, filters, edges,
# morphology, hough, color models, etc. remain UNCHANGED here)
# 👇 keep everything from your provided file as-is (no change)
# ==================================================

# --- paste all your op_bitwise_and ... op_intensity_slice_n functions here ---
# (no need to retype, they are perfect in your code above)

# ==================================================
# 🧭 Routes
# ==================================================
@app.route('/')
def home():
    try:
        return render_template('index.html')
    except Exception:
        return send_from_directory('.', 'index.html')


@app.route('/api/process', methods=['POST'])
def process_image():
    try:
        file1 = request.files.get('file')
        file2 = request.files.get('file2')
        operation = (request.form.get('operation') or '').strip().lower()

        img1 = read_image_file(file1)
        img2 = read_image_file(file2) if file2 else None

        if img1 is None:
            return jsonify({'error': 'No main image uploaded or file invalid.'}), 400

        # --- get params safely ---
        def fval(name, cast, default):
            v = request.form.get(name)
            if v in [None, '']:
                return default
            try:
                return cast(v)
            except Exception:
                return default

        # (same parameter parsing as before)
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

        # --- dispatch (keep same as your current big if/elif chain) ---
        # everything from “if operation in ['bitwise_and' …” to the end
        # stays identical to your file (no need to modify logic)

        # After your operation logic, just add safety before encoding:
        if result is None:
            return jsonify({'error': 'Operation produced no output.'}), 500
        if result.dtype != np.uint8:
            result = normalize_and_uint8(result)
        if result.ndim == 2:
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

        b64 = cv2_to_base64(result)
        meta.update({'out_shape': str(result.shape), 'out_dtype': str(result.dtype)})
        return jsonify({'image_base64': b64, 'meta': meta})

    except ValueError as ve:
        logger.warning("User error: %s", str(ve))
        return jsonify({'error': str(ve)}), 400
    except Exception as e:
        logger.exception("Unexpected server error")
        return jsonify({'error': 'Internal server error'}), 500


@app.route('/health')
def health():
    return jsonify({'status': 'ok'}), 200


# -----------------------------------
# ✅ Safe Server Entry Point
# -----------------------------------
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"Starting Flask server on port {port} (debug=False)")
    app.run(host='0.0.0.0', port=port, debug=False)
