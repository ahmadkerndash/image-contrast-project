"""
معالجة الفيديو:
    python video_main.py --input videos/sample_dark.mp4                 # توصية تلقائية
    python video_main.py --input a.mp4 --method clahe --clip 3 --side-by-side
    python video_main.py --webcam                                       # كاميرا مباشرة (q للخروج)
الطرق: auto / clahe / he / stretching / gamma
"""
import argparse
import os

from src import video


def main():
    ap = argparse.ArgumentParser(description="Video contrast enhancement")
    ap.add_argument("--input")
    ap.add_argument("--output", default=None)
    ap.add_argument("--method", default="auto",
                    choices=["auto", "clahe", "he", "stretching", "gamma"])
    ap.add_argument("--clip", type=float, default=2.0)
    ap.add_argument("--tile", type=int, default=8)
    ap.add_argument("--gamma", type=float, default=0.6)
    ap.add_argument("--color-space", default="lab", choices=["lab", "ycrcb", "hsv"])
    ap.add_argument("--side-by-side", action="store_true", help="قبل | بعد بنفس الفيديو")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--webcam", action="store_true")
    args = ap.parse_args()

    spec = None
    if args.method == "clahe":
        spec = {"method": "clahe", "params": {"clip": args.clip, "tile": args.tile}}
    elif args.method == "gamma":
        spec = {"method": "gamma", "params": {"gamma": args.gamma}}
    elif args.method in ("he", "stretching"):
        spec = {"method": args.method, "params": {}}

    if args.webcam:
        video.live_camera(spec, args.color_space)
        return
    if not args.input:
        raise SystemExit("حدد --input أو استخدم --webcam")

    out = args.output
    if out is None:
        os.makedirs("output", exist_ok=True)
        base = os.path.splitext(os.path.basename(args.input))[0]
        out = os.path.join("output", f"{base}_enhanced.mp4")

    def show(frac):
        print(f"\rالتقدّم: {frac * 100:5.1f}%", end="", flush=True)

    info = video.process_video(args.input, out, spec, args.color_space,
                               args.side_by_side, args.max_frames, show)
    print(f"\nتم. الطريقة: {info['label']}{' (تلقائية)' if info['auto'] else ''}")
    for r in info["reasons"]:
        print("  -", r)
    print(f"إطارات: {info['frames']} | FPS: {info['fps']:.1f} | الناتج: {out}")


if __name__ == "__main__":
    main()
