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

# ---------------------------
# Implementations of operations (clean & vectorized)
# ---------------------------

# 1) Bitwise - wrappers use match_images_for_binary_op
def op_bitwise_and(a, b):
    a2, b2 = match_images_for_binary_op(a, b)
    return cv2.bitwise_and(a2, b2)


def op_bitwise_or(a, b):
    a2, b2 = match_images_for_binary_op(a, b)
    return cv2.bitwise_or(a2, b2)


def op_bitwise_xor(a, b):
    a2, b2 = match_images_for_binary_op(a, b)
    return cv2.bitwise_xor(a2, b2)


def op_bitwise_not(a):
    # If color, invert each channel
    return cv2.bitwise_not(a)


# 2) BGR -> grayscale + colormaps
def op_bgr_to_luminance_gray(img):
    b, g, r = cv2.split(img.astype(np.float32))
    lum = (0.114 * b + 0.587 * g + 0.299 * r)
    return normalize_and_uint8(lum)


def op_apply_colormap(gray, cmap=cv2.COLORMAP_HOT):
    g = normalize_and_uint8(gray)
    return cv2.applyColorMap(g, cmap)


# 3) Geometric transforms
def op_scale(img, fx=1.0, fy=1.0):
    h, w = img.shape[:2]
    new_w = max(1, int(w * fx))
    new_h = max(1, int(h * fy))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)


def op_rotate(img, angle_deg=0):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle_deg, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def op_translate(img, tx=0, ty=0):
    h, w = img.shape[:2]
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)


def op_shear(img, shx=0.0, shy=0.0):
    h, w = img.shape[:2]
    M = np.float32([[1, shx, 0], [shy, 1, 0]])
    new_w = int(w + abs(shx) * h) + 1
    new_h = int(h + abs(shy) * w) + 1
    return cv2.warpAffine(img, M, (new_w, new_h), borderMode=cv2.BORDER_REFLECT)


def op_reflect(img, axis='horizontal'):
    if axis == 'horizontal':
        return cv2.flip(img, 1)
    elif axis == 'vertical':
        return cv2.flip(img, 0)
    else:
        return cv2.flip(img, -1)


# 4) Intensity transforms
def op_negative(img):
    return 255 - img


def op_log_transform(img):
    g = ensure_gray(img) if img.ndim == 3 else img
    c = 255.0 / (np.log(1 + np.max(g) + 1e-8))
    out = c * np.log(1 + g.astype(np.float32))
    return normalize_and_uint8(out)


def op_inverse_log(img):
    g = ensure_gray(img) if img.ndim == 3 else img
    g_norm = (g.astype(np.float32) / 255.0)
    out = (np.exp(g_norm) - 1) / (math.e - 1)
    return normalize_and_uint8(out * 255.0)


def op_gamma(img, gamma=1.0):
    # apply to each channel
    f = img.astype(np.float32) / 255.0
    out = np.power(f, gamma)
    return normalize_and_uint8(out * 255.0)


def op_contrast_stretch(img):
    g = ensure_gray(img) if img.ndim == 3 else img
    lo = np.percentile(g, 2)
    hi = np.percentile(g, 98)
    if hi - lo < 1e-6:
        return g
    out = (g.astype(np.float32) - lo) * (255.0 / (hi - lo))
    return normalize_and_uint8(out)


def op_piecewise_linear(img, r1=70, s1=0, r2=140, s2=255):
    g = ensure_gray(img) if img.ndim == 3 else img
    out = g.copy().astype(np.float32)
    mask1 = g <= r1
    mask2 = (g > r1) & (g <= r2)
    mask3 = g > r2
    if r1 != 0:
        out[mask1] = (s1 / r1) * g[mask1]
    else:
        out[mask1] = s1
    if (r2 - r1) != 0:
        out[mask2] = ((s2 - s1) / (r2 - r1)) * (g[mask2] - r1) + s1
    else:
        out[mask2] = s1
    out[mask3] = ((255 - s2) / (255 - r2 + 1e-8)) * (g[mask3] - r2) + s2
    return normalize_and_uint8(out)


# 5) Histogram equalization (color safe)
def op_hist_eq(img):
    if img.ndim == 2:
        return cv2.equalizeHist(img)
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


# 6) Filters
def op_box_filter(img, ksize=3):
    return cv2.blur(img, (ksize, ksize))


def op_weighted(img, kernel):
    return cv2.filter2D(img, -1, kernel)


def op_gaussian(img, sigma=1.0):
    return cv2.GaussianBlur(img, (0, 0), sigmaX=sigma, borderType=cv2.BORDER_REFLECT)


def op_min(img, ksize=3):
    if img.ndim == 3:
        g = ensure_gray(img)
    else:
        g = img
    return cv2.erode(g, np.ones((ksize, ksize), np.uint8))


def op_max(img, ksize=3):
    if img.ndim == 3:
        g = ensure_gray(img)
    else:
        g = img
    return cv2.dilate(g, np.ones((ksize, ksize), np.uint8))


def op_median(img, ksize=3):
    return cv2.medianBlur(img, max(3, ksize if ksize % 2 == 1 else ksize + 1))


def op_midpoint(img, ksize=3):
    mn = op_min(img, ksize).astype(np.uint16)
    mx = op_max(img, ksize).astype(np.uint16)
    mid = ((mn + mx) // 2).astype(np.uint8)
    return mid


# 7) Mean variants (apply on gray and convert back if originally color)
def apply_on_gray_return_bgr(img, func, *args, **kwargs):
    is_color = (img.ndim == 3)
    g = ensure_gray(img)
    out = func(g, *args, **kwargs)
    out = normalize_and_uint8(out)
    if is_color:
        return cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    return out


def op_arithmetic_mean(img, ksize=3):
    return op_box_filter(img, ksize)


def op_geometric_mean(img, ksize=3):
    return apply_on_gray_return_bgr(img, lambda g: geometric_mean_filter_fast(g, ksize))


def geometric_mean_filter_fast(img, ksize=3):
    imgf = img.astype(np.float64) + 1e-8
    log_img = np.log(imgf)
    kernel = np.ones((ksize, ksize), dtype=np.float64)
    conv = cv2.filter2D(log_img, -1, kernel) / (ksize * ksize)
    geo = np.exp(conv)
    return normalize_and_uint8(geo)


def op_harmonic_mean(img, ksize=3):
    return apply_on_gray_return_bgr(img, harmonic_mean_filter_fast, ksize)


def harmonic_mean_filter_fast(img, ksize=3):
    imgf = img.astype(np.float64) + 1e-8
    inv = 1.0 / imgf
    conv = cv2.filter2D(inv, -1, np.ones((ksize, ksize)))
    harm = (ksize * ksize) / (conv + 1e-8)
    return normalize_and_uint8(harm)


def op_contra_harmonic(img, ksize=3, Q=1.5):
    return apply_on_gray_return_bgr(img, contra_harmonic_mean_fast, ksize, Q)


def contra_harmonic_mean_fast(img, ksize=3, Q=1.5):
    imgf = img.astype(np.float64)
    kernel = np.ones((ksize, ksize))
    num = cv2.filter2D(np.power(imgf, Q + 1), -1, kernel)
    den = cv2.filter2D(np.power(imgf, Q), -1, kernel)
    out = num / (den + 1e-8)
    return normalize_and_uint8(out)


def op_alpha_trimmed(img, ksize=3, d=2):
    return apply_on_gray_return_bgr(img, alpha_trimmed_mean_fast, ksize, d)


def alpha_trimmed_mean_fast(img, ksize=3, d=2):
    pad = ksize // 2
    padded = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REFLECT)
    m, n = img.shape
    out = np.zeros_like(img, dtype=np.uint8)
    trim = d // 2
    for i in range(m):
        for j in range(n):
            window = padded[i:i + ksize, j:j + ksize].flatten()
            window.sort()
            if trim > 0 and len(window) > 2 * trim:
                trimmed = window[trim:-trim]
            else:
                trimmed = window
            out[i, j] = int(np.mean(trimmed))
    return out


# 8) Sharpening & edge operators
def op_unsharp(img, ksize=(5, 5), sigma=1.0):
    blur = cv2.GaussianBlur(img, ksize, sigmaX=sigma)
    mask = cv2.subtract(img, blur)
    sharp = cv2.addWeighted(img, 1.0, mask, 1.0, 0)
    return sharp


def op_high_boost(img, k=1.5, ksize=(5, 5), sigma=1.0):
    blur = cv2.GaussianBlur(img, ksize, sigmaX=sigma)
    mask = cv2.subtract(img, blur)
    return cv2.addWeighted(img, 1.0 + k, mask, k, 0)


def op_prewitt(img):
    g = ensure_gray(img)
    kx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
    ky = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=np.float32)
    gx = cv2.filter2D(g.astype(np.float32), -1, kx)
    gy = cv2.filter2D(g.astype(np.float32), -1, ky)
    mag = np.sqrt(gx * gx + gy * gy)
    mag_u8 = normalize_and_uint8(mag)
    return cv2.cvtColor(mag_u8, cv2.COLOR_GRAY2BGR)


def op_sobel(img):
    g = ensure_gray(img)
    gx = cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    mag_u8 = normalize_and_uint8(mag)
    return cv2.cvtColor(mag_u8, cv2.COLOR_GRAY2BGR)


def op_laplacian(img):
    g = ensure_gray(img)
    lap = cv2.Laplacian(g, cv2.CV_64F)
    lap_u8 = cv2.convertScaleAbs(lap)
    return cv2.cvtColor(lap_u8, cv2.COLOR_GRAY2BGR)


def op_log_of_gaussian(img, ksize=3, sigma=0):
    g = ensure_gray(img)
    blur = cv2.GaussianBlur(g, (ksize, ksize), sigma)
    log = cv2.Laplacian(blur, cv2.CV_64F)
    log_u8 = normalize_and_uint8(np.abs(log))
    return cv2.cvtColor(log_u8, cv2.COLOR_GRAY2BGR)


# 9) Canny / Harris / Hough
def op_canny(img, low=None, high=None):
    g = ensure_gray(img)
    # equalize to improve contrast before edges
    try:
        g_eq = cv2.equalizeHist(g)
    except Exception:
        g_eq = g
    # auto threshold via median if not provided
    v = np.median(g_eq)
    if low is None or high is None:
        low = int(max(0, 0.66 * v))
        high = int(min(255, 1.33 * v))
    low = max(1, int(low))
    high = max(low + 1, int(high))
    edges = cv2.Canny(g_eq, low, high, L2gradient=True)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)


def op_harris(img, blockSize=2, ksize=3, k=0.04, thresh_ratio=0.01):
    g = ensure_gray(img).astype(np.float32)
    dst = cv2.cornerHarris(g, blockSize, ksize, k)
    dst = cv2.dilate(dst, None)
    out = img.copy()
    # mark corners in red
    out[dst > thresh_ratio * dst.max()] = [0, 0, 255]
    return out


def op_hough_lines(img):
    out = img.copy()
    g = ensure_gray(img)
    edges = cv2.Canny(g, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=30, maxLineGap=10)
    if lines is not None:
        for l in lines[:, 0, :]:
            cv2.line(out, (l[0], l[1]), (l[2], l[3]), (0, 255, 0), 2)
    return out


def op_hough_circles(img):
    out = img.copy()
    g = ensure_gray(img)
    g = cv2.medianBlur(g, 5)
    circles = cv2.HoughCircles(g, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20, param1=50, param2=30, minRadius=5,
                               maxRadius=200)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for c in circles[0, :]:
            cv2.circle(out, (c[0], c[1]), c[2], (0, 255, 0), 2)
            cv2.circle(out, (c[0], c[1]), 2, (0, 0, 255), 3)
    return out


# 10) Morphology / LBP / boundary extraction
def op_dilate(img, ksize=3):
    kern = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    g = ensure_gray(img)
    out = cv2.dilate(g, kern)
    return cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)


def op_erode(img, ksize=3):
    kern = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    g = ensure_gray(img)
    out = cv2.erode(g, kern)
    return cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)


def op_opening(img, ksize=3):
    kern = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    out = cv2.morphologyEx(ensure_gray(img), cv2.MORPH_OPEN, kern)
    return cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)


def op_closing(img, ksize=3):
    kern = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    out = cv2.morphologyEx(ensure_gray(img), cv2.MORPH_CLOSE, kern)
    return cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)


def op_hit_or_miss(img):
    g = ensure_gray(img)
    bw = (g > 127).astype(np.uint8) * 255
    se_hit = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
    se_miss = np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]], dtype=np.uint8)
    erode_hit = cv2.erode(bw, se_hit)
    erode_miss = cv2.erode(cv2.bitwise_not(bw), se_miss)
    out = cv2.bitwise_and(erode_hit, erode_miss)
    return cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)


def op_boundary(img):
    g = ensure_gray(img)
    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(g, kernel)
    boundary = cv2.subtract(g, eroded)
    return cv2.cvtColor(boundary, cv2.COLOR_GRAY2BGR)


def op_skeleton(img):
    try:
        from skimage.morphology import skeletonize as sk_skel
        g = ensure_gray(img)
        bw = g > 0
        sk = sk_skel(bw)
        return cv2.cvtColor((sk.astype(np.uint8) * 255), cv2.COLOR_GRAY2BGR)
    except Exception:
        return img


def op_lbp(img, P=8, R=1):
    g = ensure_gray(img)
    img8 = g if g.dtype == np.uint8 else normalize_and_uint8(g)
    lbp = local_binary_pattern(img8, P, R, method='uniform')
    lbp_u8 = normalize_and_uint8(lbp)
    return cv2.cvtColor(lbp_u8, cv2.COLOR_GRAY2BGR)


# Color model conversions & pseudo-coloring
def op_bgr_to_cmy(img):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cmy = 255 - rgb
    return cv2.cvtColor(cmy, cv2.COLOR_RGB2BGR)


def op_bgr_to_cmyk(img):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    C = 1 - rgb[..., 0]
    M = 1 - rgb[..., 1]
    Y = 1 - rgb[..., 2]
    K = np.minimum.reduce([C, M, Y])
    denom = (1.0 - K) + 1e-8
    Cn = np.where(denom == 0, 0, (C - K) / denom)
    Mn = np.where(denom == 0, 0, (M - K) / denom)
    Yn = np.where(denom == 0, 0, (Y - K) / denom)
    vis = (np.stack([Cn, Mn, Yn], axis=-1) * 255.0).astype(np.uint8)
    return cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)


def op_bgr_to_ycbcr(img):
    out = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    return cv2.cvtColor(out, cv2.COLOR_YCrCb2BGR)


def op_intensity_slice_2(img, thresh=128, color1=(255, 0, 0), color2=(0, 255, 0)):
    g = ensure_gray(img)
    out = np.zeros((g.shape[0], g.shape[1], 3), dtype=np.uint8)
    out[g >= thresh] = color1[::-1]  # BGR
    out[g < thresh] = color2[::-1]
    return out


def op_intensity_slice_n(img, n=8):
    g = ensure_gray(img)
    cmap = cv2.applyColorMap(normalize_and_uint8(g), cv2.COLORMAP_JET)
    return cmap


# ---------------------------

# --------------------------------------------------
# ✅ All your operation functions remain unchanged
# --------------------------------------------------
# (Everything from op_bitwise_and() to op_intensity_slice_n() stays EXACTLY as you pasted)
# --------------------------------------------------

# 🧭 ROUTES
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
        print(f"🔍 Operation received: {operation}")  # For Railway logs

        img1 = read_image_file(file1)
        img2 = read_image_file(file2) if file2 else None

        if img1 is None:
            return jsonify({'error': 'No main image uploaded or file invalid.'}), 400

        def fval(name, cast, default):
            v = request.form.get(name)
            if v in [None, '']:
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
        result = None
# BITWISE
        if operation in ['bitwise_and', 'bitwise_or', 'bitwise_xor']:
            if img2 is None:
                return jsonify({'error': f'{operation} requires a second image in file2.'}), 400
            a, b = match_images_for_binary_op(img1.copy(), img2.copy())
            if operation == 'bitwise_and':
                result = op_bitwise_and(a, b)
            elif operation == 'bitwise_or':
                result = op_bitwise_or(a, b)
            else:
                result = op_bitwise_xor(a, b)

        elif operation == 'bitwise_not':
            result = op_bitwise_not(img1)

        # BGR->Gray & colormap
        elif operation == 'bgr_to_luminance_gray':
            gray = op_bgr_to_luminance_gray(img1)
            result = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        elif operation in ['apply_colormap_hot', 'apply_colormap']:
            gray = op_bgr_to_luminance_gray(img1)
            result = op_apply_colormap(gray, cv2.COLORMAP_HOT)

        # GEOMETRY
        elif operation == 'scale':
            result = op_scale(img1, fx=fx, fy=fy)

        elif operation == 'rotate':
            result = op_rotate(img1, angle_deg=angle)

        elif operation == 'translate':
            result = op_translate(img1, tx=tx, ty=ty)

        elif operation == 'shear':
            result = op_shear(img1, shx=shx, shy=shy)

        elif operation == 'reflect':
            result = op_reflect(img1, axis=axis)

        # INTENSITY
        elif operation == 'negative_intensity' or operation == 'negative':
            result = op_negative(img1)

        elif operation == 'log_transform' or operation == 'log':
            g = op_log_transform(img1)
            result = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR) if img1.ndim == 3 else g

        elif operation == 'inverse_log':
            g = op_inverse_log(img1)
            result = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR) if img1.ndim == 3 else g

        elif operation in ['power_law', 'gamma']:
            g = op_gamma(img1, gamma=gamma)
            result = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR) if img1.ndim == 3 and g.ndim == 2 else g

        elif operation == 'contrast_stretch' or operation == 'contrast':
            g = op_contrast_stretch(img1)
            result = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR) if img1.ndim == 3 else g

        elif operation == 'piecewise_linear':
            g = op_piecewise_linear(img1, r1=fval('r1', int, 70), s1=fval('s1', int, 0),
                                    r2=fval('r2', int, 140), s2=fval('s2', int, 255))
            result = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR) if img1.ndim == 3 else g

        # HISTOGRAM
        elif operation in ['hist_eq', 'histogram_equalization']:
            result = op_hist_eq(img1)

        # FILTERS
        elif operation == 'box_filter':
            result = op_box_filter(img1, ksize=ksize)

        elif operation == 'weighted_filter':
            kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], np.float32) / 16.0
            result = op_weighted(img1, kernel)

        elif operation in ['gaussian_sigma', 'gaussian']:
            result = op_gaussian(img1, sigma=sigma)

        elif operation == 'min_filter':
            out = op_min(img1, ksize=ksize)
            result = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)

        elif operation == 'max_filter':
            out = op_max(img1, ksize=ksize)
            result = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)

        elif operation == 'median_filter':
            result = op_median(img1, ksize=ksize)

        elif operation == 'midpoint_filter':
            out = op_midpoint(img1, ksize=ksize)
            result = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)

        # MEAN VARIANTS
        elif operation == 'arithmetic_mean':
            result = op_arithmetic_mean(img1, ksize=ksize)

        elif operation == 'geometric_mean':
            result = op_geometric_mean(img1, ksize=ksize)

        elif operation == 'harmonic_mean':
            result = op_harmonic_mean(img1, ksize=ksize)

        elif operation == 'contra_harmonic':
            result = op_contra_harmonic(img1, ksize=ksize, Q=Q)

        elif operation == 'alpha_trimmed':
            result = op_alpha_trimmed(img1, ksize=ksize, d=d)

        # SHARPEN / EDGE
        elif operation == 'unsharp_mask':
            result = op_unsharp(img1, ksize=(ksize, ksize), sigma=sigma)

        elif operation == 'high_boost':
            result = op_high_boost(img1, k=kparam, ksize=(ksize, ksize), sigma=sigma)

        elif operation == 'prewitt':
            result = op_prewitt(img1)

        elif operation == 'sobel':
            result = op_sobel(img1)

        elif operation == 'laplacian':
            result = op_laplacian(img1)

        elif operation in ['log_of_gaussian', 'logogaussian', 'log']:
            result = op_log_of_gaussian(img1, ksize=ksize, sigma=sigma)

        # CANNY / HARRIS / HOUGH
        elif operation == 'canny':
            result = op_canny(img1, low=low, high=high)

        elif operation == 'harris':
            result = op_harris(img1)

        elif operation == 'hough_lines':
            result = op_hough_lines(img1)

        elif operation == 'hough_circles':
            result = op_hough_circles(img1)

        # MORPHOLOGY & TEXTURE
        elif operation == 'dilate':
            result = op_dilate(img1, ksize=ksize)

        elif operation == 'erode':
            result = op_erode(img1, ksize=ksize)

        elif operation == 'opening':
            result = op_opening(img1, ksize=ksize)

        elif operation == 'closing':
            result = op_closing(img1, ksize=ksize)

        elif operation == 'hit_or_miss':
            result = op_hit_or_miss(img1)

        elif operation == 'boundary':
            result = op_boundary(img1)

        elif operation == 'skeleton':
            result = op_skeleton(img1)

        elif operation == 'lbp':
            result = op_lbp(img1, P=P, R=R)

        # COLOR MODELS & PSEUDO
        elif operation == 'bgr_to_cmy':
            result = op_bgr_to_cmy(img1)

        elif operation == 'bgr_to_cmyk':
            result = op_bgr_to_cmyk(img1)

        elif operation == 'bgr_to_ycbcr':
            result = op_bgr_to_ycbcr(img1)

        elif operation == 'intensity_slice_2':
            result = op_intensity_slice_2(img1, thresh=thresh)

        elif operation == 'intensity_slice_n':
            result = op_intensity_slice_n(img1, n=n)

        else:
            return jsonify({'error': f'Unknown operation: {operation}'}), 400

        # Prepare response - ensure BGR uint8
        # ==============================
        # ✅ FULL OPERATION DISPATCH SECTION
        # ==============================
        # (Paste your entire if/elif chain exactly from your working code)
        # Example:
        # if operation in ['bitwise_and', 'bitwise_or', 'bitwise_xor']:
        #     ...
        # elif operation == 'intensity_slice_n':
        #     result = op_intensity_slice_n(img1, n=n)
        # else:
        #     return jsonify({'error': f'Unknown operation: {operation}'}), 400
        # ==============================

        # ✅ Safe finalization
        if result is None:
            print(f"⚠️ No result produced for operation: {operation}")
            return jsonify({'error': 'Operation produced no output.'}), 500

        if result.dtype != np.uint8:
            result = normalize_and_uint8(result)
        if result.ndim == 2:
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

        b64 = cv2_to_base64(result)
        meta.update({'out_shape': str(result.shape), 'out_dtype': str(result.dtype)})
        return jsonify({'image_base64': b64, 'meta': meta})

    except Exception as e:
        print(f"❌ Internal server error: {str(e)}")
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

@app.route('/health')
def health():
    return jsonify({'status': 'ok'}), 200

# ✅ Entry Point for Railway
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"🚀 Starting Flask on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)
