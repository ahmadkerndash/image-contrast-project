"""
طرق تحسين التباين (Contrast Enhancement)
كل دالة بتشتغل على صورة رمادية (2D, uint8).
والدالة apply_to_color بتطبّق أي طريقة على الصور الملونة بدون تشويه الألوان.
"""
import cv2
import numpy as np


# ---------- 1) Contrast Stretching ----------
def contrast_stretching(gray, low_pct=1, high_pct=99):
    """
    نمدّ المدى بين percentile منخفض وعالي (بدل min/max)
    عشان نقاط الضجيج المتطرفة ما تخرب النتيجة.
    s = (r - low) / (high - low) * 255
    """
    low, high = np.percentile(gray, (low_pct, high_pct))
    if high == low:
        return gray.copy()
    out = (gray.astype(np.float32) - low) / (high - low) * 255.0
    return np.clip(out, 0, 255).astype(np.uint8)


# ---------- 2) Histogram Equalization ----------
def histogram_equalization(gray):
    """
    HE: نستخدم CDF (التوزيع التراكمي) كدالة تحويل.
    (مكتوبة يدوياً للفهم، وبتعطي نتيجة قريبة جداً من cv2.equalizeHist)
    """
    hist = np.bincount(gray.ravel(), minlength=256)
    cdf = hist.cumsum()
    cdf_min = cdf[cdf > 0][0]
    n = gray.size
    if n == cdf_min:
        return gray.copy()
    lut = np.round((cdf - cdf_min) / (n - cdf_min) * 255).astype(np.uint8)
    return lut[gray]


# ---------- 3) CLAHE ----------
def clahe(gray, clip_limit=2.0, tile_grid=(8, 8)):
    """
    Contrast Limited Adaptive HE:
    - بيقسم الصورة لمربعات صغيرة (tiles) ويعمل HE لكل مربع.
    - clip_limit بيحدّ تضخيم الضجيج.
    """
    c = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    return c.apply(gray)


# ---------- 4) Gamma Correction ----------
def gamma_correction(gray, gamma=0.7):
    """
    s = 255 * (r/255)^gamma
    gamma < 1 : بتفتّح الصورة   |   gamma > 1 : بتغمّقها
    """
    lut = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)],
                   dtype=np.uint8)
    return lut[gray]


# ---------- الصور الملونة ----------
def apply_to_color(bgr, method, space="lab"):
    """
    نحوّل الصورة لفضاء لوني بيفصل الإضاءة عن اللون،
    نحسّن قناة الإضاءة فقط، ونرجّعها لـ BGR.
    space: 'lab' (قناة L) أو 'ycrcb' (قناة Y) أو 'hsv' (قناة V)
    """
    if space == "lab":
        conv, back, ch = cv2.COLOR_BGR2LAB, cv2.COLOR_LAB2BGR, 0
    elif space == "ycrcb":
        conv, back, ch = cv2.COLOR_BGR2YCrCb, cv2.COLOR_YCrCb2BGR, 0
    elif space == "hsv":
        conv, back, ch = cv2.COLOR_BGR2HSV, cv2.COLOR_HSV2BGR, 2
    else:
        raise ValueError("space must be lab / ycrcb / hsv")

    converted = cv2.cvtColor(bgr, conv)
    converted[:, :, ch] = method(np.ascontiguousarray(converted[:, :, ch]))
    return cv2.cvtColor(converted, back)


def naive_rgb_he(bgr):
    """HE على كل قناة RGB لحالها (للمقارنة فقط: بتشوّه الألوان!)"""
    return cv2.merge([histogram_equalization(c) for c in cv2.split(bgr)])


def apply_any(img, fn, space="lab"):
    """يطبّق دالة التحسين على صورة رمادية أو ملونة (حسب نوعها)."""
    if img.ndim == 2:
        return fn(img)
    return apply_to_color(img, fn, space)
