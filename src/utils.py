import os
import cv2
import matplotlib
matplotlib.use("Agg")  # يحفظ الصور بدون ما يفتح نافذة
import matplotlib.pyplot as plt

IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def list_images(folder):
    return sorted(f for f in os.listdir(folder) if f.lower().endswith(IMG_EXT))


def is_grayscale(bgr):
    """بنعتبر الصورة رمادية إذا القنوات الثلاثة متطابقة (صور الأشعة غالباً هيك)."""
    if bgr.ndim == 2:
        return True
    b, g, r = cv2.split(bgr)
    return bool((b == g).all() and (g == r).all())


def luminance(img):
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def save_comparison(results, out_path, title):
    """
    results: dict {اسم الطريقة: صورة}
    بيرسم صف للصور وصف للهيستوغرامات ويحفظهم بملف واحد.
    """
    n = len(results)
    fig, axes = plt.subplots(2, n, figsize=(3.2 * n, 6.4))
    for i, (name, img) in enumerate(results.items()):
        ax = axes[0, i]
        if img.ndim == 2:
            ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        else:
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(name, fontsize=10)
        ax.axis("off")

        axh = axes[1, i]
        axh.hist(luminance(img).ravel(), bins=256, range=(0, 255), color="steelblue")
        axh.set_xlim(0, 255)
        axh.set_xlabel("Intensity")
        if i == 0:
            axh.set_ylabel("Pixels")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
