"""
اختبار الضجيج:
    python noise_test.py --image input/sample_gray_lowcontrast.png
    python noise_test.py --image input/xxx.png --kind saltpepper --levels 0.01 0.03 0.05
"""
import argparse
import csv
import os

import cv2

from src import recommend, noise


def main():
    ap = argparse.ArgumentParser(description="Noise robustness test")
    ap.add_argument("--image", required=True)
    ap.add_argument("--kind", default="gaussian", choices=["gaussian", "saltpepper"])
    ap.add_argument("--levels", type=float, nargs="+", default=None,
                    help="gaussian: sigma (مثلاً 5 10 20) | saltpepper: نسبة (مثلاً 0.01 0.03)")
    ap.add_argument("--color-space", default="lab", choices=["lab", "ycrcb", "hsv"])
    ap.add_argument("--output", default="output")
    args = ap.parse_args()

    levels = args.levels or ([5, 10, 20, 30] if args.kind == "gaussian"
                             else [0.01, 0.03, 0.05, 0.08])
    img = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"ما قدرت أقرأ الصورة: {args.image}")
    if recommend.analyze(img)["is_gray"]:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    rec = recommend.recommend(img)
    methods = {label: recommend.build_fn({"method": m, "params": p})
               for label, m, p in [
                   ("Stretching", "stretching", {}),
                   ("HE", "he", {}),
                   ("CLAHE", "clahe", {"clip": 2.0, "tile": 8}),
                   ("Gamma 0.6", "gamma", {"gamma": 0.6}),
               ]}
    auto_name = "Auto: " + recommend.label(rec)
    methods[auto_name] = recommend.build_fn(rec)

    rows, _ = noise.run_noise_test(img, methods, args.kind, levels, args.color_space)

    os.makedirs(args.output, exist_ok=True)
    name = os.path.splitext(os.path.basename(args.image))[0]
    csv_path = os.path.join(args.output, f"noise_{name}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["level", "method", "PSNR", "SSIM", "Amplification", "Contrast"])
        w.writeheader()
        w.writerows(rows)
    png_path = os.path.join(args.output, f"noise_{name}.png")
    noise.plot_noise_results(rows, png_path, f"{name} ({args.kind})")

    print(f"{'level':>7} {'method':<34}{'PSNR':>7}{'SSIM':>8}{'Amp':>7}{'RMS':>8}")
    for r in rows:
        print(f"{r['level']:>7} {r['method']:<34}{r['PSNR']:>7}{r['SSIM']:>8}{r['Amplification']:>7}{r['Contrast']:>8}")
    print(f"\nالرسم: {png_path}\nالجدول: {csv_path}")


if __name__ == "__main__":
    main()
