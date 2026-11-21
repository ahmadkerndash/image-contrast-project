"""
مقاييس تقييم جودة التباين (بدون صورة مرجعية)
"""
import cv2
import numpy as np


def _to_gray(img):
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def rms_contrast(img):
    """RMS Contrast = الانحراف المعياري للإضاءة. أعلى = تباين أكبر."""
    return float(_to_gray(img).std())


def entropy(img):
    """
    Shannon Entropy: كمية المعلومات. أعلى = تفاصيل أكتر
    (القيمة القصوى النظرية = 8 للصورة 8-bit).
    """
    g = _to_gray(img)
    p = np.bincount(g.ravel(), minlength=256) / g.size
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def ambe(original, enhanced):
    """
    AMBE: الفرق المطلق بين متوسط الإضاءة قبل وبعد.
    أقل = حافظنا على السطوع الأصلي أكتر.
    """
    return float(abs(_to_gray(original).mean() - _to_gray(enhanced).mean()))


def evaluate(original, enhanced):
    return {
        "RMS Contrast": round(rms_contrast(enhanced), 2),
        "Entropy": round(entropy(enhanced), 3),
        "AMBE": round(ambe(original, enhanced), 2),
    }
