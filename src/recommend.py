"""
التوصية التلقائية: بنحلّل الصورة (سطوع، مدى الإضاءة، ضجيج، نوع)
وبنختار أنسب طريقة مع شرح السبب.
"""
import cv2
import numpy as np

from . import enhance


def estimate_noise(gray):
    """تقدير انحراف الضجيج (طريقة Immerkaer): قيمة أعلى = ضجيج أكتر."""
    g = gray.astype(np.float32)
    h, w = g.shape
    if h < 4 or w < 4:
        return 0.0
    kernel = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32)
    conv = cv2.filter2D(g, -1, kernel)[1:-1, 1:-1]
    return float(np.abs(conv).sum() * np.sqrt(0.5 * np.pi) / (6.0 * (w - 2) * (h - 2)))


def analyze(img):
    """بيرجع إحصائيات الصورة."""
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    is_gray = img.ndim == 2 or bool(
        (img[:, :, 0] == img[:, :, 1]).all() and (img[:, :, 1] == img[:, :, 2]).all()
    )
    p1, p99 = np.percentile(gray, (1, 99))
    return {
        "is_gray": is_gray,
        "mean": float(gray.mean()),
        "std": float(gray.std()),
        "range": float(p99 - p1),
        "dark_ratio": float((gray < 50).mean()),
        "bright_ratio": float((gray > 200).mean()),
        "noise": estimate_noise(gray),
    }


def recommend(img):
    """
    بيرجع: {"method": ..., "params": {...}, "reasons": [...], "stats": {...}}
    method أحد: clahe / he / stretching / gamma
    """
    s = analyze(img)
    reasons = []
    noisy = s["noise"] > 8

    if s["is_gray"]:
        reasons.append("الصورة رمادية (غالباً طبية).")
    else:
        reasons.append("الصورة ملونة: رح نحسّن قناة الإضاءة فقط لحماية الألوان.")

    if noisy:
        reasons.append(f"الضجيج مرتفع (≈{s['noise']:.1f}): رح نقلّل clip limit عشان ما نضخّمه.")

    # 1) مدى الإضاءة ضيّق (صورة باهتة)
    if s["range"] < 110:
        reasons.append(f"مدى الإضاءة ضيّق ({s['range']:.0f} من 255): الهيستوغرام متجمّع.")
        if not s["is_gray"] and not noisy:
            method, params = "stretching", {"low": 1, "high": 99}
            reasons.append("Contrast Stretching كافي ومناسب: بيمدّ المدى بدون ما يغيّر شكل الصورة.")
        else:
            method, params = "clahe", {"clip": 1.5 if noisy else 3.0, "tile": 8}
            reasons.append("CLAHE أنسب: بيحسّن التباين محلياً مع حدّ لتضخيم الضجيج.")

    # 2) صورة غامقة
    elif s["mean"] < 85:
        reasons.append(f"الصورة غامقة (متوسط الإضاءة {s['mean']:.0f}).")
        if s["is_gray"]:
            method, params = "clahe", {"clip": 1.5 if noisy else 3.0, "tile": 8}
            reasons.append("CLAHE بيطلّع تفاصيل المناطق الغامقة بالصور الطبية.")
        else:
            gamma = float(np.clip(np.log(0.5) / np.log(max(s["mean"], 1) / 255.0), 0.35, 0.9))
            method, params = "gamma", {"gamma": round(gamma, 2)}
            reasons.append(f"Gamma < 1 (محسوبة = {gamma:.2f}) بتفتّح الغامق وبتحافظ على الطبيعي.")

    # 3) صورة فاتحة
    elif s["mean"] > 170:
        reasons.append(f"الصورة فاتحة (متوسط الإضاءة {s['mean']:.0f}).")
        gamma = float(np.clip(np.log(0.5) / np.log(min(s["mean"], 254) / 255.0), 1.1, 2.0))
        method, params = "gamma", {"gamma": round(gamma, 2)}
        reasons.append(f"Gamma > 1 (محسوبة = {gamma:.2f}) بتغمّق الصورة وبترجّع التفاصيل.")

    # 4) الحالة العادية
    else:
        method, params = "clahe", {"clip": 1.5 if noisy else 2.0, "tile": 8}
        reasons.append("التباين والسطوع متوسطين: CLAHE خيار آمن وبيعطي نتيجة متوازنة.")

    return {"method": method, "params": params, "reasons": reasons, "stats": s}


def build_fn(spec):
    """بيحوّل {"method":..., "params":...} لدالة تحسين جاهزة (gray -> gray)."""
    m, p = spec["method"], spec.get("params", {})
    if m == "clahe":
        t = int(p.get("tile", 8))
        return lambda g: enhance.clahe(g, float(p.get("clip", 2.0)), (t, t))
    if m == "he":
        return enhance.histogram_equalization
    if m == "stretching":
        return lambda g: enhance.contrast_stretching(g, p.get("low", 1), p.get("high", 99))
    if m == "gamma":
        return lambda g: enhance.gamma_correction(g, float(p.get("gamma", 0.6)))
    raise ValueError(f"طريقة غير معروفة: {m}")


METHOD_LABELS = {
    "clahe": "CLAHE",
    "he": "Histogram Equalization",
    "stretching": "Contrast Stretching",
    "gamma": "Gamma",
}


def label(spec):
    p = spec.get("params", {})
    extra = ", ".join(f"{k}={v}" for k, v in p.items())
    return METHOD_LABELS[spec["method"]] + (f" ({extra})" if extra else "")
