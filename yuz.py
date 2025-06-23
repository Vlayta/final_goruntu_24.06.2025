import cv2
import mediapipe as mp
import numpy as np
import random

# MediaPipe'in yüz mesh modülü
mp_yuz_mesh = mp.solutions.face_mesh
mp_cizim = mp.solutions.drawing_utils

# Kare boyutu ve kare sayısını belirleyen değişkenler
kare_boyutu = 10  # Her mozaik karesinin kenar uzunluğu (piksel)
mozaik_sikligi = 1.0  # 1.0 tam mozaik, 0.0 hiç mozaik değil

# Kamerayı başlat
kamera = cv2.VideoCapture(0)

with mp_yuz_mesh.FaceMesh(
    max_num_faces=1,  # Aynı anda en fazla 1 yüz analiz edilsin
    refine_landmarks=True,  # Göz bebeği gibi detaylı noktalar dahil olsun
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as yuz:

    while kamera.isOpened():
        basarili, kare = kamera.read()
        if not basarili:
            break

        # Görüntüyü BGR'den RGB'ye çevir
        rgb_kare = cv2.cvtColor(kare, cv2.COLOR_BGR2RGB)

        # Yüz tespiti yap
        sonuc = yuz.process(rgb_kare)

        # Eğer yüz bulunduysa
        if sonuc.multi_face_landmarks:
            for yuz_noktasi in sonuc.multi_face_landmarks:
                # Tüm yüz noktalarının piksel koordinatlarını al
                noktalar = []
                for nokta in yuz_noktasi.landmark:
                    x = int(nokta.x * kare.shape[1])
                    y = int(nokta.y * kare.shape[0])
                    noktalar.append((x, y))

                # Tüm noktaların sınırlayıcı kutusunu al (yüz bölgesi)
                x_koordinatlar = [n[0] for n in noktalar]
                y_koordinatlar = [n[1] for n in noktalar]
                x_min = max(min(x_koordinatlar), 0)
                y_min = max(min(y_koordinatlar), 0)
                x_max = min(max(x_koordinatlar), kare.shape[1])
                y_max = min(max(y_koordinatlar), kare.shape[0])

                # Yüz alanını al
                yuz_alani = kare[y_min:y_max, x_min:x_max].copy()

                # Mozaik uygulanacak yüzey
                for y in range(0, yuz_alani.shape[0], kare_boyutu):
                    for x in range(0, yuz_alani.shape[1], kare_boyutu):
                        # Kare bölge sınırları
                        y1 = y
                        y2 = min(y + kare_boyutu, yuz_alani.shape[0])
                        x1 = x
                        x2 = min(x + kare_boyutu, yuz_alani.shape[1])

                        # Rastgele bir renk üret (ya da ortalama renk de alınabilir)
                        renk = yuz_alani[y1:y2, x1:x2].mean(axis=(0, 1)).astype(np.uint8)

                        # Belirli bir ihtimalle mozaik uygula (rastgelelik)
                        if random.random() < mozaik_sikligi:
                            yuz_alani[y1:y2, x1:x2] = renk

                # Mozaiklenmiş yüzü ana kareye yerleştir
                kare[y_min:y_max, x_min:x_max] = yuz_alani

        # Görüntüyü ekranda göster
        cv2.imshow("Mozaiklenmis Yuz", kare)

        # ESC tuşuna basıldığında çık
        if cv2.waitKey(1) & 0xFF == 27:
            break

kamera.release()
cv2.destroyAllWindows()
