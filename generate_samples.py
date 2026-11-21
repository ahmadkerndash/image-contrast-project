"""
بيولّد صور تجريبية جوّا مجلد input/ (إذا ما عندك صور جاهزة):
  - sample_color_dark.png   : صورة طبيعية ملونة غامقة + تباين ضعيف
  - sample_color_hazy.png   : صورة طبيعية ملونة باهتة (ضبابية)
  - sample_gray_lowcontrast.png : صورة رمادية بتباين ضعيف (للتجربة)
بعدين حط صورك الطبية الحقيقية (أشعة...) بنفس المجلد.
"""
import os
import cv2
import numpy as np
from skimage import data

os.makedirs("input", exist_ok=True)

# صورة ملونة غامقة
astro = cv2.cvtColor(data.astronaut(), cv2.COLOR_RGB2BGR)
dark = (255 * (astro / 255.0) ** 2.2 * 0.8).astype(np.uint8)
cv2.imwrite("input/sample_color_dark.png", dark)

# صورة ملونة باهتة (تباين مضغوط حول المنتصف)
coffee = cv2.cvtColor(data.coffee(), cv2.COLOR_RGB2BGR)
hazy = (coffee.astype(np.float32) * 0.35 + 110).astype(np.uint8)
cv2.imwrite("input/sample_color_hazy.png", hazy)

# صورة رمادية بتباين ضعيف
cam = data.camera()
low = (cam.astype(np.float32) * 0.3 + 90).astype(np.uint8)
cv2.imwrite("input/sample_gray_lowcontrast.png", low)

print("تم توليد الصور بمجلد input/")

# فيديو تجريبي غامق (كاميرا بتتحرك فوق الصورة)
os.makedirs("videos", exist_ok=True)
big = cv2.resize(dark, (640, 640))
writer = cv2.VideoWriter("videos/sample_dark.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 20, (360, 360))
for i in range(60):
    x = int(i * (640 - 360) / 59)
    y = int(i * 120 / 59)
    writer.write(big[y:y + 360, x:x + 360])
writer.release()
print("تم توليد فيديو تجريبي: videos/sample_dark.mp4")
