"""
مشروع تحسين تباين الصور - مادة نظم الوسائط
الاستخدام:
    python main.py                       # كل الصور بمجلد input/
    python main.py --color-space ycrcb   # اختيار الفضاء اللوني للصور الملونة
    python main.py --clip 3.0 --tile 8   # ضبط CLAHE
    python main.py --gamma 0.6           # ضبط Gamma
"""
import argparse
import csv
import os

import cv2

from src import enhance, metrics, recommend
from src.utils import list_images, is_grayscale, save_comparison


def process_image(path, args, out_dir):
    name = os.path.splitext(os.path.basename(path))[0]
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        print(f"  ! ما قدرت أقرأ: {path}")
        return []

    gray_mode = is_grayscale(bgr)
    base = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if gray_mode else bgr
    kind = "grayscale/medical" if gray_mode else f"color ({args.color_space})"
    print(f"- {name}: {kind}")

    # الطرق مع المعاملات اللي اختارها المستخدم
    methods = {
        "Stretching": enhance.contrast_stretching,
        "HE": enhance.histogram_equalization,
        "CLAHE": lambda g: enhance.clahe(g, args.clip, (args.tile, args.tile)),
        f"Gamma {args.gamma}": lambda g: enhance.gamma_correction(g, args.gamma),
    }

    results = {"Original": base}
    for mname, fn in methods.items():
        if gray_mode:
            results[mname] = fn(base)
        else:
            results[mname] = enhance.apply_to_color(base, fn, args.color_space)

    # التوصية التلقائية
    if args.auto:
        rec = recommend.recommend(base)
        print("  توصية: " + recommend.label(rec))
        for r in rec["reasons"]:
            print("    -", r)
        fn = recommend.build_fn(rec)
        key = "Auto: " + recommend.label(rec)
        results[key] = fn(base) if gray_mode else enhance.apply_to_color(base, fn, args.color_space)

    # للصور الملونة: نضيف HE الساذج على RGB لنوضّح مشكلة تشوّه الألوان
    if not gray_mode:
        results["HE on RGB (naive)"] = enhance.naive_rgb_he(base)

    # حفظ كل نتيجة + الصورة المقارنة
    for mname, img in results.items():
        safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in mname.lower())
        cv2.imwrite(os.path.join(out_dir, f"{name}_{safe}.png"), img)
    save_comparison(results, os.path.join(out_dir, f"{name}_comparison.png"), name)

    # المقاييس
    rows = []
    for mname, img in results.items():
        m = metrics.evaluate(base, img)
        rows.append({"image": name, "method": mname, **m})
    return rows


def main():
    ap = argparse.ArgumentParser(description="Image Contrast Enhancement")
    ap.add_argument("--input", default="input")
    ap.add_argument("--output", default="output")
    ap.add_argument("--color-space", default="lab", choices=["lab", "ycrcb", "hsv"])
    ap.add_argument("--clip", type=float, default=2.0, help="CLAHE clip limit")
    ap.add_argument("--tile", type=int, default=8, help="CLAHE tile grid size")
    ap.add_argument("--gamma", type=float, default=0.6)
    ap.add_argument("--auto", action="store_true", help="أضف نتيجة التوصية التلقائية")
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)
    files = list_images(args.input)
    if not files:
        print("ما في صور بمجلد input/ ... شغّل: python generate_samples.py")
        return

    all_rows = []
    for f in files:
        all_rows += process_image(os.path.join(args.input, f), args, args.output)

    # جدول المقاييس CSV + طباعة
    csv_path = os.path.join(args.output, "metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["image", "method", "RMS Contrast", "Entropy", "AMBE"])
        w.writeheader()
        w.writerows(all_rows)

    print("\n{:<28}{:<20}{:>8}{:>10}{:>8}".format("image", "method", "RMS", "Entropy", "AMBE"))
    for r in all_rows:
        print("{:<28}{:<20}{:>8}{:>10}{:>8}".format(
            r["image"], r["method"], r["RMS Contrast"], r["Entropy"], r["AMBE"]))
    print(f"\nالنتائج بمجلد: {args.output}/  (والمقاييس: metrics.csv)")


if __name__ == "__main__":
    main()
