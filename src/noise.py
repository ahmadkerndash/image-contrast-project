"""
اختبار الضجيج: بنضيف ضجيج للصورة ونقيس أي طريقة أقوى وأقل تضخيماً للضجيج.
المرجع = نتيجة الطريقة على الصورة النظيفة.
  PSNR / SSIM  : أعلى = النتيجة بالضجيج أقرب لنتيجتها النظيفة (أكثر ثباتاً)
  Amplification: ضجيج الخرج / ضجيج الدخل (أقل = أفضل، 1 = ما ضخّمنا)
  Contrast     : RMS contrast للخرج (لازم تقرأه مع PSNR: ثبات عالي مع تباين ضعيف مش تحسين حقيقي)
"""
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from . import enhance
from .recommend import estimate_noise


def add_gaussian(img, sigma, seed=0):
    rng = np.random.default_rng(seed)
    noisy = img.astype(np.float32) + rng.normal(0, sigma, img.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def add_salt_pepper(img, amount, seed=0):
    """amount = نسبة البكسلات المتأثرة (مثلاً 0.02 = 2%)."""
    rng = np.random.default_rng(seed)
    out = img.copy()
    mask = rng.random(img.shape[:2])
    out[mask < amount / 2] = 0
    out[mask > 1 - amount / 2] = 255
    return out


def add_noise(img, kind, level, seed=0):
    if kind == "gaussian":
        return add_gaussian(img, level, seed)
    if kind == "saltpepper":
        return add_salt_pepper(img, level, seed)
    raise ValueError("kind must be gaussian / saltpepper")


def _lum(img):
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def run_noise_test(base, methods, kind="gaussian", levels=(5, 10, 20, 30),
                   space="lab", seed=0):
    """
    methods: dict {اسم: دالة gray->gray}
    بيرجع (rows, previews):
      rows     : قائمة نتائج [{level, method, PSNR, SSIM, Amplification, Contrast}]
      previews : صور للعرض {"noisy": {...}, "outputs": {...}} عند أعلى مستوى ضجيج
    """
    # مراجع نظيفة
    clean_out = {"No enhancement": base}
    for name, fn in methods.items():
        clean_out[name] = enhance.apply_any(base, fn, space)

    rows, previews = [], {}
    for level in levels:
        noisy = add_noise(base, kind, level, seed)
        sigma_in = max(estimate_noise(_lum(noisy)), 1e-6)

        outs = {"No enhancement": noisy}
        for name, fn in methods.items():
            outs[name] = enhance.apply_any(noisy, fn, space)

        for name, out in outs.items():
            ref = clean_out[name]
            psnr = peak_signal_noise_ratio(_lum(ref), _lum(out), data_range=255)
            ssim = structural_similarity(_lum(ref), _lum(out), data_range=255)
            amp = estimate_noise(_lum(out)) / sigma_in
            rows.append({
                "level": level, "method": name,
                "PSNR": round(float(min(psnr, 99.0)), 2),
                "SSIM": round(float(ssim), 3),
                "Amplification": round(float(amp), 2),
                "Contrast": round(float(_lum(out).std()), 1),
            })
        if level == max(levels):
            previews = {"noisy": noisy, "outputs": outs}
    return rows, previews


def plot_noise_results(rows, out_path, title="Noise robustness"):
    """يرسم PSNR و SSIM و Amplification حسب مستوى الضجيج لكل طريقة."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = make_noise_figure(rows, title)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def make_noise_figure(rows, title="Noise robustness"):
    import matplotlib.pyplot as plt

    methods = list(dict.fromkeys(r["method"] for r in rows))
    levels = sorted({r["level"] for r in rows})
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, key, ylabel in zip(axes, ["PSNR", "SSIM", "Amplification"],
                               ["PSNR (dB) - higher better", "SSIM - higher better",
                                "Noise amplification - lower better"]):
        for m in methods:
            ys = [next(r[key] for r in rows if r["method"] == m and r["level"] == lv)
                  for lv in levels]
            ax.plot(levels, ys, marker="o", label=m)
        ax.set_xlabel("Noise level")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    return fig
