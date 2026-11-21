"""
واجهة مشروع تحسين تباين الصور - مادة نظم الوسائط
التشغيل:   streamlit run app.py
التبويبات: صورة (قبل/بعد) | اختبار الضجيج | فيديو
"""
import os
import tempfile

import cv2
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from src import enhance, metrics, noise, recommend, video
from src.utils import luminance

st.set_page_config(page_title="تحسين تباين الصور", layout="wide")
st.title("🖼️ تحسين تباين الصور")
st.caption("مادة نظم الوسائط | صور (طبيعية/طبية) + اختبار الضجيج + فيديو")


# ====================== دوال مساعدة ======================
def show(img, caption=None):
    if img.ndim == 2:
        st.image(img, width="stretch", clamp=True, caption=caption)
    else:
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), width="stretch", caption=caption)


def hist_fig(img, color):
    fig, ax = plt.subplots(figsize=(4, 2.2))
    ax.hist(luminance(img).ravel(), bins=256, range=(0, 255), color=color)
    ax.set_xlim(0, 255)
    ax.set_xlabel("Intensity")
    ax.set_ylabel("Pixels")
    fig.tight_layout()
    return fig


def read_upload(uploaded):
    data = np.frombuffer(uploaded.getvalue(), np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)  # بتشتغل حتى لو الاسم عربي


# ====================== الشريط الجانبي ======================
st.sidebar.header("الإعدادات")
uploaded = st.sidebar.file_uploader(
    "ارفع صورة", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"]
)
color_space = st.sidebar.selectbox("الفضاء اللوني (للملون)", ["lab", "ycrcb", "hsv"])
mode = st.sidebar.radio("اختيار الطريقة", ["🤖 توصية تلقائية", "✋ يدوي"])

manual_spec = None
if mode == "✋ يدوي":
    mname = st.sidebar.selectbox("الطريقة", list(recommend.METHOD_LABELS.values()))
    key = {v: k for k, v in recommend.METHOD_LABELS.items()}[mname]
    params = {}
    if key == "clahe":
        params = {
            "clip": st.sidebar.slider("Clip limit", 1.0, 10.0, 2.0, 0.5),
            "tile": st.sidebar.slider("Tile size", 2, 16, 8),
        }
    elif key == "gamma":
        params = {"gamma": st.sidebar.slider("Gamma", 0.2, 3.0, 0.6, 0.05)}
    elif key == "stretching":
        lo, hi = st.sidebar.slider("Percentiles", 0, 100, (1, 99))
        params = {"low": lo, "high": hi}
    manual_spec = {"method": key, "params": params}

base = None
if uploaded is not None:
    bgr = read_upload(uploaded)
    if bgr is None:
        st.sidebar.error("ما قدرت أقرأ الصورة. جرّب png أو jpg.")
    else:
        stats = recommend.analyze(bgr)
        base = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if stats["is_gray"] else bgr

tab_img, tab_noise, tab_vid = st.tabs(["🖼️ صورة (قبل / بعد)", "🔊 اختبار الضجيج", "🎞️ فيديو"])

# ====================== تبويب الصورة ======================
with tab_img:
    if base is None:
        st.info("👈 ارفع صورة من الشريط الجانبي لتبدأ.")
    else:
        gray_mode = base.ndim == 2
        if mode == "🤖 توصية تلقائية":
            rec = recommend.recommend(base)
            spec = {"method": rec["method"], "params": rec["params"]}
            st.success(f"التوصية التلقائية: **{recommend.label(spec)}**")
            with st.expander("ليش هالطريقة؟ (تحليل الصورة)", expanded=True):
                for r in rec["reasons"]:
                    st.write("•", r)
                s = rec["stats"]
                c = st.columns(5)
                c[0].metric("السطوع (mean)", f"{s['mean']:.0f}")
                c[1].metric("مدى الإضاءة", f"{s['range']:.0f}")
                c[2].metric("الانحراف (std)", f"{s['std']:.0f}")
                c[3].metric("الضجيج", f"{s['noise']:.1f}")
                c[4].metric("النوع", "رمادية" if s["is_gray"] else "ملونة")
        else:
            spec = manual_spec

        fn = recommend.build_fn(spec)
        result = enhance.apply_any(base, fn, color_space)
        st.write("نوع الصورة:", "رمادية / طبية" if gray_mode else f"ملونة ({color_space})")

        sub1, sub2 = st.tabs(["جنب بعض", "شريط تمرير (Slider)"])
        with sub1:
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("قبل (Original)")
                show(base)
                st.pyplot(hist_fig(base, "gray"))
            with c2:
                st.subheader(f"بعد ({recommend.label(spec)})")
                show(result)
                st.pyplot(hist_fig(result, "steelblue"))
        with sub2:
            st.caption("يسار = قبل، يمين = بعد")
            pos = st.slider("الموضع", 0, 100, 50, key="pos")
            h, w = base.shape[:2]
            a = base if base.ndim == 3 else cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
            b = result if result.ndim == 3 else cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
            x = int(w * pos / 100)
            combo = np.concatenate([a[:, :x], b[:, x:]], axis=1)
            if 0 < x < w:
                combo[:, max(x - 1, 0):x + 1] = (0, 255, 255)
            show(combo)

        st.subheader("المقاييس")
        m0, m1 = metrics.evaluate(base, base), metrics.evaluate(base, result)
        st.table({
            "المقياس": ["RMS Contrast (أعلى = أفضل)", "Entropy (أعلى = تفاصيل أكتر)",
                        "AMBE (أقل = حافظ على السطوع)"],
            "قبل": [m0["RMS Contrast"], m0["Entropy"], m0["AMBE"]],
            "بعد": [m1["RMS Contrast"], m1["Entropy"], m1["AMBE"]],
        })

        with st.expander("قارن كل الطرق مرة وحدة"):
            all_specs = {
                "Stretching": {"method": "stretching", "params": {}},
                "HE": {"method": "he", "params": {}},
                "CLAHE": {"method": "clahe", "params": {"clip": 2.0, "tile": 8}},
                "Gamma 0.6": {"method": "gamma", "params": {"gamma": 0.6}},
            }
            cols = st.columns(len(all_specs))
            for col, (name, sp) in zip(cols, all_specs.items()):
                with col:
                    show(enhance.apply_any(base, recommend.build_fn(sp), color_space), name)

        ok, buf = cv2.imencode(".png", result)
        if ok:
            st.download_button("⬇️ تحميل الصورة المحسّنة", buf.tobytes(),
                               file_name="enhanced.png", mime="image/png")

# ====================== تبويب اختبار الضجيج ======================
with tab_noise:
    st.write("بنضيف ضجيج للصورة ونشوف أي طريقة **أثبت** وأقل تضخيماً للضجيج.")
    if base is None:
        st.info("👈 ارفع صورة من الشريط الجانبي.")
    else:
        c1, c2 = st.columns(2)
        kind_label = c1.selectbox("نوع الضجيج", ["Gaussian", "Salt & Pepper"])
        kind = "gaussian" if kind_label == "Gaussian" else "saltpepper"
        if kind == "gaussian":
            levels = c2.multiselect("مستويات الضجيج (sigma)", [5, 10, 15, 20, 30, 40],
                                    default=[5, 10, 20, 30])
        else:
            levels = c2.multiselect("نسبة الضجيج", [0.01, 0.02, 0.03, 0.05, 0.08, 0.1],
                                    default=[0.01, 0.03, 0.05, 0.08])

        if st.button("▶️ شغّل الاختبار") and levels:
            rec = recommend.recommend(base)
            methods = {
                "Stretching": recommend.build_fn({"method": "stretching", "params": {}}),
                "HE": recommend.build_fn({"method": "he", "params": {}}),
                "CLAHE": recommend.build_fn({"method": "clahe", "params": {"clip": 2.0, "tile": 8}}),
                "Gamma 0.6": recommend.build_fn({"method": "gamma", "params": {"gamma": 0.6}}),
                "Auto: " + recommend.label(rec): recommend.build_fn(rec),
            }
            with st.spinner("عم يحسب..."):
                rows, prev = noise.run_noise_test(base, methods, kind, sorted(levels), color_space)

            st.pyplot(noise.make_noise_figure(rows, f"{kind_label} noise"))

            st.subheader(f"الصور عند أعلى مستوى ضجيج ({max(levels)})")
            cols = st.columns(len(prev["outputs"]))
            for col, (name, img) in zip(cols, prev["outputs"].items()):
                with col:
                    show(img, "Noisy input" if name == "No enhancement" else name)

            st.subheader("الجدول")
            st.dataframe(rows, width="stretch")
            st.info(
                "كيف تقرأ النتائج: **PSNR/SSIM** أعلى = أثبت بوجود الضجيج. "
                "**Amplification** أقل = ضخّم الضجيج أقل. "
                "بس لا تنسى عمود **Contrast**: طريقة ثباتها عالي لكن تباينها ضعيف "
                "(مثل Gamma) ما بتحسّن التباين فعلياً. المطلوب توازن."
            )

# ====================== تبويب الفيديو ======================
with tab_vid:
    st.write("ارفع فيديو، وبنحسّن تباين كل إطار ونرجّعلك فيديو جديد.")
    vfile = st.file_uploader("ارفع فيديو", type=["mp4", "avi", "mov", "mkv"], key="vid")
    side = st.checkbox("فيديو واحد: قبل | بعد (جنب بعض)", value=True)
    limit = st.number_input("أقصى عدد إطارات (0 = كامل الفيديو)", 0, 100000, 300, 50)

    if vfile is not None and st.button("▶️ عالج الفيديو"):
        suffix = os.path.splitext(vfile.name)[1] or ".mp4"
        tmp_in = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp_in.write(vfile.getvalue())
        tmp_in.close()
        tmp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        bar = st.progress(0.0, text="عم يعالج الإطارات...")
        try:
            vspec = None if mode == "🤖 توصية تلقائية" else manual_spec
            info = video.process_video(
                tmp_in.name, tmp_out, vspec, color_space, side,
                int(limit) or None, lambda f: bar.progress(f, text="عم يعالج الإطارات...")
            )
            bar.empty()
            st.success(
                f"تم! الطريقة: **{info['label']}**"
                f"{' (تلقائية)' if info['auto'] else ''} | "
                f"{info['frames']} إطار | {info['fps']:.0f} FPS | "
                f"{info['size'][0]}×{info['size'][1]}"
            )
            for r in info["reasons"]:
                st.write("•", r)

            st.subheader("عيّنات من الإطارات (قبل / بعد)")
            for before, after in info["samples"]:
                c1, c2 = st.columns(2)
                with c1:
                    show(before, "قبل")
                with c2:
                    show(after, "بعد")

            with open(tmp_out, "rb") as fh:
                st.download_button("⬇️ تحميل الفيديو الناتج", fh.read(),
                                   file_name="enhanced.mp4", mime="video/mp4")
            st.caption("إذا الفيديو ما اشتغل بمشغّل المتصفح، حمّله وشغّله بأي مشغّل (VLC).")
        except Exception as e:
            bar.empty()
            st.error(f"صار خطأ: {e}")
        finally:
            for pth in (tmp_in.name, tmp_out):
                try:
                    os.remove(pth)
                except OSError:
                    pass
