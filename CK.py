import cv2
import cvzone
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import random
import time
import math

detector = HandDetector(detectionCon=0.8, maxHands=1)
classifier = Classifier("converted_keras/keras_model.h5", "converted_keras/labels.txt")
labels = ["M", "TM", "SM"]

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
checkpoint_radius = 80
checkpoint_x = random.randint(50, 600)
checkpoint_y = random.randint(50, 400)
skor = 0
maks_waktu = 60 
waktu_mulai = time.time()

# Fungsi untuk memberikan perintah acak
def perintah_acak():
    perintah = ["menggenggam", "setengah menggenggam", "tidak menggenggam"]
    return random.choice(perintah)

# Fungsi untuk mengecek apakah tangan berada di posisi checkpoint
def tangan_di_checkpoint(frame, checkpoint_x, checkpoint_y, checkpoint_radius):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 60, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) < 500:
            continue

        (x, y, w, h) = cv2.boundingRect(contour)
        center_x, center_y = x + w // 2, y + h // 2

        jarak = math.sqrt((center_x - checkpoint_x) ** 2 + (center_y - checkpoint_y) ** 2)
        if jarak < checkpoint_radius:
            return True

    return False

perintah_saat_ini = perintah_acak()
cv2.namedWindow('Permainan Genggaman Tangan', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Permainan Genggaman Tangan', 1280, 720)

# Loop utama
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Prediksi status tangan menggunakan model Keras
    status = classifier.predict(frame)

    # Periksa apakah tangan berada di checkpoint
    if tangan_di_checkpoint(frame, checkpoint_x, checkpoint_y, checkpoint_radius):
        # Periksa apakah status tangan sesuai dengan perintah
        if status == perintah_saat_ini:
            # Tambahkan skor dan pindahkan checkpoint
            skor += 1
            checkpoint_x = random.randint(50, 600)
            checkpoint_y = random.randint(50, 400)
            perintah_saat_ini = perintah_acak()

    cv2.circle(frame, (checkpoint_x, checkpoint_y), checkpoint_radius, (0, 255, 0), 2)
    waktu_tersisa = maks_waktu - (time.time() - waktu_mulai)
    cv2.putText(frame, f"Perintah: {perintah_saat_ini}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Skor: {skor}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Waktu: {int(waktu_tersisa)}", (frame.shape[1] - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Permainan Genggaman Tangan', frame)

    # Keluar dari loop jika ditekan 'q' atau waktu habis
    if cv2.waitKey(1) & 0xFF == ord('q') or waktu_tersisa <= 0:
        break

# Setelah keluar dari loop utama, tambahkan kode berikut untuk menampilkan skor akhir di tengah layar
frame = cv2.putText(frame, f"Permainan selesai! Skor akhir Anda adalah {skor}.", 
                    (int(frame.shape[1]/2) - 250, int(frame.shape[0]/2)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4)

# Tampilkan frame terakhir
cv2.imshow('Permainan Genggaman Tangan', frame)
cv2.waitKey(0)  # Tunggu tombol apa pun ditekan

# Bersihkan
cap.release()
cv2.destroyAllWindows()

print(f"Permainan selesai! Skor akhir Anda adalah {skor}.")
