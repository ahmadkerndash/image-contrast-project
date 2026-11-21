# مشروع تحسين تباين الصور (Image Contrast Enhancement)
مادة: نظم الوسائط | اللغة: Python

## الطرق المنفّذة
| الطريقة | الفكرة | مناسبة لـ |
|---|---|---|
| Contrast Stretching | مدّ المدى بين percentile 1% و99% | صور باهتة |
| Histogram Equalization | توزيع الإضاءة بالـ CDF | تحسين عام وسريع |
| CLAHE | HE محلي مع حدّ للتضخيم | الصور الطبية |
| Gamma Correction | تحويل غير خطي | صور غامقة/فاتحة |

**الصور الملونة:** بنحسّن قناة الإضاءة فقط (LAB أو YCrCb أو HSV) عشان ما تتشوّه الألوان.
بنضيف كمان "HE on RGB (naive)" للمقارنة وإثبات المشكلة.

## المقاييس
- **RMS Contrast:** أعلى = تباين أكبر
- **Entropy:** أعلى = معلومات/تفاصيل أكتر (الحد الأقصى 8)
- **AMBE:** أقل = حافظنا على السطوع الأصلي أكتر

## التشغيل
```bash
python -m venv venv
venv\Scripts\activate          # ويندوز   (ماك/لينكس: source venv/bin/activate)
pip install -r requirements.txt

python generate_samples.py     # اختياري: صور تجريبية
python main.py                 # بيعالج كل الصور بمجلد input/
```

خيارات إضافية:
```bash
python main.py --color-space ycrcb
python main.py --clip 3.0 --tile 8
python main.py --gamma 0.5
```

## المخرجات (مجلد output/)
- `<اسم>_comparison.png`: مقارنة الصور + الهيستوغرامات (للتقرير)
- `<اسم>_<الطريقة>.png`: كل نتيجة لحالها
- `metrics.csv`: جدول المقاييس

## هيكل المشروع
```
main.py              البرنامج الرئيسي
generate_samples.py  توليد صور تجريبية
src/enhance.py       خوارزميات التحسين
src/metrics.py       مقاييس التقييم
src/utils.py         قراءة/رسم/حفظ
input/  output/
```

## الميزات الإضافية

### 1) التوصية التلقائية 🤖
بتحلّل الصورة (سطوع، مدى الإضاءة، ضجيج، رمادية/ملونة) وبتختار الطريقة مع شرح السبب.
| الحالة | القرار |
|---|---|
| مدى ضيّق + ملونة + ضجيج قليل | Contrast Stretching |
| مدى ضيّق + رمادية أو ضجيج عالي | CLAHE (clip أقل لو الضجيج عالي) |
| غامقة + ملونة | Gamma < 1 (بتنحسب من متوسط الإضاءة) |
| غامقة + رمادية (طبية) | CLAHE |
| فاتحة | Gamma > 1 |
| عادية | CLAHE |
```bash
python main.py --auto
```

### 2) اختبار الضجيج 🔊
بنضيف ضجيج (Gaussian أو Salt & Pepper) وبنقيس ثبات كل طريقة:
PSNR/SSIM (أعلى = أثبت)، Amplification (أقل = ضخّم الضجيج أقل)، Contrast (RMS).
```bash
python noise_test.py --image input/sample_gray_lowcontrast.png
python noise_test.py --image input/xxx.png --kind saltpepper --levels 0.01 0.03 0.05
```
النتيجة: `output/noise_<اسم>.png` (رسومات) و`.csv`.
ملاحظة: اقرأ PSNR مع عمود Contrast. طريقة ثباتها عالي وتباينها ضعيف (مثل Gamma) ما بتحسّن التباين فعلياً.

### 3) معالجة الفيديو 🎞️
```bash
python video_main.py --input videos/sample_dark.mp4 --side-by-side
python video_main.py --input a.mp4 --method clahe --clip 3
python video_main.py --webcam          # كاميرا مباشرة، q للخروج
```
الطرق: auto / clahe / he / stretching / gamma. بالوضع التلقائي التوصية بتنحسب من أول إطار.
الناتج: `output/<اسم>_enhanced.mp4`.
(الفيديو المحفوظ بصيغة mp4v: إذا ما اشتغل بالمتصفح شغّله بـ VLC.)

## الواجهة الرسومية
```bash
pip install -r requirements.txt
streamlit run app.py
```
3 تبويبات: صورة (قبل/بعد + شريط تمرير + مقاييس) | اختبار الضجيج | فيديو.

## هيكل المشروع
```
main.py              معالجة صور من input/ (مع --auto)
noise_test.py        اختبار الضجيج
video_main.py        معالجة فيديو / كاميرا
app.py               الواجهة (Streamlit)
generate_samples.py  صور وفيديو تجريبي
src/enhance.py       الخوارزميات
src/recommend.py     التوصية التلقائية
src/noise.py         اختبار الضجيج
src/video.py         معالجة الفيديو
src/metrics.py       المقاييس
src/utils.py         أدوات مساعدة
input/  videos/  output/
```
