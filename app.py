from flask import Flask, render_template, request, jsonify, send_from_directory
import cv2
import numpy as np
import base64
import math
from skimage.feature import local_binary_pattern
from skimage import util
import io
import os
from collections import Counter
import heapq

# HOG
try:
    from skimage.feature import hog as sk_hog
    from skimage import exposure
except Exception:
    sk_hog = None
    exposure = None

app = Flask(__name__, static_folder='static', template_folder='templates')

# Helpers

def cv2_to_base64(img):
    """Encode OpenCV BGR image (uint8) to base64 PNG string."""
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
    # Remove alpha if present
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
    """
    Make shapes/channels/dtypes compatible for bitwise operations.
    Returns tuple (a2, b2).
    """
    if a is None or b is None:
        return a, b
    # Resize b to a's size if different
    if a.shape[:2] != b.shape[:2]:
        b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)
    # If different channel counts, convert grayscale->BGR
    if a.ndim != b.ndim:
        if a.ndim == 2:
            a = cv2.cvtColor(a, cv2.COLOR_GRAY2BGR)
        if b.ndim == 2:
            b = cv2.cvtColor(b, cv2.COLOR_GRAY2BGR)
    # Match dtype (convert to common - prefer uint8)
    if a.dtype != b.dtype:
        b = b.astype(a.dtype)
    return a, b

# Basic 

def normalize_and_uint8(img):
    """Normalize float images to 0-255 uint8."""
    if img is None:
        return None
    if img.dtype == np.uint8:
        return img
    mn = float(np.min(img))
    mx = float(np.max(img))
    if mx - mn <= 1e-8:
        return np.clip(img, 0, 255).astype(np.uint8)
    scaled = (img - mn) * 255.0 / (mx - mn)
    return np.clip(scaled, 0, 255).astype(np.uint8)

# Implementations 

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

# Compression, HOG

# RLE (text output)
def op_rle_text(img):
    """Run-Length Encoding over grayscale bytes. Returns text string."""
    g = ensure_gray(img)
    flattened = g.flatten().tolist()
    if len(flattened) == 0:
        return "", ""
    out_pairs = []
    prev = flattened[0]
    count = 1
    for v in flattened[1:]:
        if v == prev:
            count += 1
        else:
            out_pairs.append(f"{prev}:{count}")
            prev = v
            count = 1
    out_pairs.append(f"{prev}:{count}")
    # return text representation
    rle_text = " ".join(out_pairs)
    # For UI consistency return an image too (original) and rle_text in meta
    return g, rle_text


# Huffman
class HuffmanNode:
    def __init__(self, freq, symbol=None, left=None, right=None):
        self.freq = freq
        self.symbol = symbol
        self.left = left
        self.right = right

    def __lt__(self, other):
        return self.freq < other.freq


def build_huffman_code(freq_map):
    heap = []
    for sym, freq in freq_map.items():
        heapq.heappush(heap, HuffmanNode(freq, symbol=sym))
    if len(heap) == 0:
        return {}
    # edge case: single symbol
    if len(heap) == 1:
        node = heapq.heappop(heap)
        return {node.symbol: "0"}
    while len(heap) > 1:
        n1 = heapq.heappop(heap)
        n2 = heapq.heappop(heap)
        merged = HuffmanNode(n1.freq + n2.freq, left=n1, right=n2)
        heapq.heappush(heap, merged)
    root = heapq.heappop(heap)

    codes = {}

    def traverse(node, prefix=""):
        if node.symbol is not None:
            codes[node.symbol] = prefix or "0"
            return
        traverse(node.left, prefix + "0")
        traverse(node.right, prefix + "1")

    traverse(root, "")
    return codes


def op_huffman_visualize(img, max_bits_visual=5_000_000):
    g = ensure_gray(img)
    flat = g.flatten().tolist()
    freq = Counter(flat)
    codes = build_huffman_code(freq)
    if not codes:
        return g, {}, 0, 0
    # encode
    bits = "".join(codes[p] for p in flat)
    nbits = len(bits)
    # protect extremely large outputs by truncating visualization
    if nbits > max_bits_visual:
        bits = bits[:max_bits_visual]
        nbits = len(bits)
    # pack bits into rows to make a square-ish binary image
    w = int(math.ceil(math.sqrt(nbits)))
    h = int(math.ceil(nbits / w))
    bin_img = np.zeros((h, w), dtype=np.uint8)
    for i, bit in enumerate(bits):
        r = i // w
        c = i % w
        bin_img[r, c] = 255 if bit == "1" else 0
    vis = cv2.cvtColor(bin_img, cv2.COLOR_GRAY2BGR)
    return vis, codes, nbits, len(flat) * 8


# Zig-zag helpers (for DCT blocks)
def zigzag_indices(n):
    idx = []
    for s in range(2 * n - 1):
        if s % 2 == 0:
            # even sum - go down
            for i in range(s + 1):
                j = s - i
                if i < n and j < n:
                    idx.append((i, j))
        else:
            # odd sum - go up
            for j in range(s + 1):
                i = s - j
                if i < n and j < n:
                    idx.append((i, j))
    return idx[:n * n]


ZIGZAG_CACHE = {k: zigzag_indices(k) for k in [4, 8, 16]}


def zigzag_flatten(block):
    n = block.shape[0]
    idx = ZIGZAG_CACHE.get(n, zigzag_indices(n))
    return np.array([block[i, j] for (i, j) in idx])


def zigzag_restore(flat, n):
    idx = ZIGZAG_CACHE.get(n, zigzag_indices(n))
    block = np.zeros((n, n), dtype=flat.dtype)
    for k, (i, j) in enumerate(idx):
        if k < len(flat):
            block[i, j] = flat[k]
    return block


# block DCT compress using zig-zag keep first K coefficients per block
def op_dct_block_compress(img, block_size=8, keep=10):
    """Perform block-wise DCT on Y channel, keep first 'keep' zig-zag coefficients per block."""
    img_bgr = ensure_bgr(img)
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    Y = ycrcb[:, :, 0]
    h, w = Y.shape
    # pad to multiple of block_size
    pad_h = (block_size - (h % block_size)) % block_size
    pad_w = (block_size - (w % block_size)) % block_size
    Yp = np.pad(Y, ((0, pad_h), (0, pad_w)), mode='reflect')
    H, W = Yp.shape
    outY = np.zeros_like(Yp)
    for i in range(0, H, block_size):
        for j in range(0, W, block_size):
            block = Yp[i:i + block_size, j:j + block_size]
            block = block - 128.0
            d = cv2.dct(block)
            flat = zigzag_flatten(d)
            # zero out after keep
            flat[keep:] = 0
            d2 = zigzag_restore(flat, block_size)
            idct = cv2.idct(d2) + 128.0
            outY[i:i + block_size, j:j + block_size] = idct
    # crop back
    outY = outY[:h, :w]
    ycrcb[:, :, 0] = outY
    out = cv2.cvtColor(ycrcb.astype(np.uint8), cv2.COLOR_YCrCb2BGR)
    return out


# Uniform quantization (levels)
def op_uniform_quantize(img, levels=8):
    g = ensure_gray(img)
    # map to [0, levels-1], then back to 0..255
    q = np.floor((g.astype(np.float32) * (levels - 1) / 255.0) + 0.5)
    recon = (q * (255.0 / (levels - 1))).astype(np.uint8)
    # return BGR for consistency
    return cv2.cvtColor(recon, cv2.COLOR_GRAY2BGR)


# Color compression via KMeans (k)
def op_color_quant_kmeans(img, k=8):
    # use cv2.kmeans on pixel color space
    Z = img.reshape((-1, 3)).astype(np.float32)
    # criteria, attempts
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.1)
    attempts = 2
    _, labels, centers = cv2.kmeans(Z, k, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
    centers = centers.astype(np.uint8)
    quant = centers[labels.flatten()].reshape(img.shape)
    return quant


# Shift (translation) with user params shift_x, shift_y
def op_shift(img, shift_x=0, shift_y=0):
    h, w = img.shape[:2]
    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)


# HOG visualization
def op_hog_visualize(img, pixels_per_cell=(8, 8), cells_per_block=(2, 2), orientations=9):
    if sk_hog is None:
        # fallback: return grayscale if skimage not installed
        return cv2.cvtColor(ensure_gray(img), cv2.COLOR_GRAY2BGR)
    g = ensure_gray(img)
    hog_img, hog_vis = sk_hog(g, orientations=orientations, pixels_per_cell=pixels_per_cell,
                              cells_per_block=cells_per_block, visualize=True, feature_vector=False)
    # skimage returns an image scaled; rescale for display
    vis = exposure.rescale_intensity(hog_vis, in_range=(0, np.max(hog_vis)))
    vis_u8 = normalize_and_uint8((vis * 255.0))
    return cv2.cvtColor(vis_u8, cv2.COLOR_GRAY2BGR)

# SIFT Feature Detector

def try_sift():
    """Create a SIFT detector safely."""
    try:
        return cv2.SIFT_create()                   # OpenCV 4.5+
    except:
        try:
            return cv2.xfeatures2d.SIFT_create()   # For opencv-contrib
        except:
            return None

def op_sift_features(img):
    """Detect SIFT keypoints and visualize."""
    sift = try_sift()
    if sift is None:
        return cv2.cvtColor(ensure_gray(img), cv2.COLOR_GRAY2BGR), 0

    gray = ensure_gray(img)
    keypoints, descriptors = sift.detectAndCompute(gray, None)

    vis = cv2.drawKeypoints(
        img,
        keypoints,
        None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        color=(0, 255, 0)
    )
    return vis, len(keypoints)

# Route 
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

        # fetch params with safe defaults
        def fval(name, cast, default):
            v = request.form.get(name)
            if v is None or v == '':
                return default
            try:
                return cast(v)
            except Exception:
                return default

        # New/used params
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

        # New params for added features
        rle_text_flag = request.form.get('rle_flag', '0')  # optional
        # Huffman nothing extra
        dct_block_size = int(max(2, fval('dct_block_size', int, 8)))
        dct_keep = int(max(1, fval('dct_keep', int, 10)))
        quant_levels = int(max(2, fval('quant_levels', int, 8)))
        color_k = int(max(1, fval('color_k', int, 8)))
        shift_x = fval('shift_x', int, 0)
        shift_y = fval('shift_y', int, 0)
        hog_ppc = int(max(1, fval('hog_ppc', int, 8)))
        hog_cpb = int(max(1, fval('hog_cpb', int, 2)))
        hog_orient = int(max(1, fval('hog_orient', int, 9)))

        meta = {'shape': str(img1.shape), 'dtype': str(img1.dtype), 'operation': operation}

        result = None  # default

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

        elif operation == 'rle_text':
            # returns original image and RLE text in meta['rle']
            gray, rle_text = op_rle_text(img1)
            result = cv2.cvtColor(normalize_and_uint8(gray), cv2.COLOR_GRAY2BGR)
            meta['rle'] = (rle_text[:5000] + '...') if len(rle_text) > 5000 else rle_text
            meta['rle_len_pairs'] = len(rle_text.split()) if rle_text else 0

        elif operation == 'huffman_encoded_image':
            vis, codes, nbits, original_bits = op_huffman_visualize(img1)
            result = vis
            meta['huffman_codes'] = {str(k): v for k, v in list(codes.items())[:50]}  # limit listing
            meta['encoded_bits'] = nbits
            meta['orig_bits'] = original_bits

        elif operation == 'dct_block':
            result = op_dct_block_compress(img1, block_size=dct_block_size, keep=dct_keep)
            meta['dct_block_size'] = dct_block_size
            meta['dct_keep'] = dct_keep

        elif operation == 'quantize_levels':
            result = op_uniform_quantize(img1, levels=quant_levels)
            meta['quant_levels'] = quant_levels

        elif operation == 'zigzag_dct':
            # perform DCT+zigzag keeping dct_keep coefficients
            result = op_dct_block_compress(img1, block_size=dct_block_size, keep=dct_keep)
            meta['zigzag_block'] = dct_block_size
            meta['zigzag_keep'] = dct_keep

        elif operation == 'color_compress_kmeans':
            result = op_color_quant_kmeans(img1, k=color_k)
            meta['color_k'] = color_k

        elif operation == 'shift':
            result = op_shift(img1, shift_x=shift_x, shift_y=shift_y)
            meta['shift_x'] = shift_x
            meta['shift_y'] = shift_y

        elif operation == 'hog_visualize':
            result = op_hog_visualize(img1, pixels_per_cell=(hog_ppc, hog_ppc),
                                      cells_per_block=(hog_cpb, hog_cpb), orientations=hog_orient)
            meta['hog_ppc'] = hog_ppc
            meta['hog_cpb'] = hog_cpb
            meta['hog_orient'] = hog_orient

        elif operation == 'sift_features':
            result, count = op_sift_features(img1)
            meta['sift_keypoints'] = count

        else:
            return jsonify({'error': f'Unknown operation: {operation}'}), 400

        # Prepare response - ensure BGR uint8
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
        return jsonify({'error': 'Internal server error', 'detail': str(e)}), 500

@app.route('/health')
def health():
    return jsonify({'status': 'ok'})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
