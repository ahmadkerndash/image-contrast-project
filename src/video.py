"""
معالجة الفيديو: بنحسّن تباين كل إطار (Frame) ونحفظ فيديو جديد.
"""
import cv2
import numpy as np

from . import enhance
from .recommend import recommend, build_fn, label


def _label_frame(img, text):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (len(text) * 11 + 12, 26), (0, 0, 0), -1)
    cv2.putText(out, text, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA)
    return out


def enhance_frame(frame, fn, space="lab"):
    return enhance.apply_any(frame, fn, space)


def process_video(in_path, out_path, spec=None, space="lab",
                  side_by_side=False, max_frames=None, progress=None):
    """
    spec = None  -> توصية تلقائية من أول إطار
    spec = {"method":..., "params":...} -> طريقة يدوية
    progress(frac) اختياري لعرض شريط التقدّم.
    بيرجع dict فيه معلومات المعالجة.
    """
    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        raise IOError(f"ما قدرت أفتح الفيديو: {in_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps != fps or fps < 1:
        fps = 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if max_frames:
        total = min(total, max_frames) if total else max_frames

    ok, first = cap.read()
    if not ok:
        cap.release()
        raise IOError("الفيديو فاضي أو ما قدرت أقرأ منه.")

    auto = spec is None
    if auto:
        rec = recommend(first)
        spec = {"method": rec["method"], "params": rec["params"]}
        reasons = rec["reasons"]
    else:
        reasons = []
    fn = build_fn(spec)

    out_w = w * 2 if side_by_side else w
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, h))
    if not writer.isOpened():
        cap.release()
        raise IOError("ما قدرت أنشئ ملف الفيديو الناتج.")

    n = 0
    samples = {}
    frame = first
    while ok:
        res = enhance_frame(frame, fn, space)
        if side_by_side:
            out = np.hstack([_label_frame(frame, "Before"), _label_frame(res, "After")])
        else:
            out = res
        writer.write(out)
        if n in (0, max(total // 2, 1), max(total - 1, 1)) and n not in samples:
            samples[n] = (frame.copy(), res.copy())
        n += 1
        if progress and total:
            progress(min(n / total, 1.0))
        if max_frames and n >= max_frames:
            break
        ok, frame = cap.read()

    cap.release()
    writer.release()
    return {
        "frames": n, "fps": fps, "size": (w, h), "spec": spec,
        "auto": auto, "reasons": reasons, "label": label(spec),
        "samples": list(samples.values()),
    }


def live_camera(spec=None, space="lab", camera_id=0):
    """
    عرض مباشر من الكاميرا (قبل | بعد). اضغط q للخروج.
    spec = None -> توصية تلقائية تتحدّث كل 60 إطار.
    """
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise IOError("ما قدرت أفتح الكاميرا.")
    auto = spec is None
    fn, n, name = None, 0, ""
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if auto and n % 60 == 0:
            rec = recommend(frame)
            fn = build_fn(rec)
            name = label(rec)
        elif not auto and fn is None:
            fn, name = build_fn(spec), label(spec)
        res = enhance_frame(frame, fn, space)
        view = np.hstack([_label_frame(frame, "Before"), _label_frame(res, name)])
        cv2.imshow("Live contrast enhancement (q = quit)", view)
        n += 1
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
