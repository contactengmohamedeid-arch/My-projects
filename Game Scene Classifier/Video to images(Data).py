import cv2
import os

# اسم اللعبة اللي هنعملها الفولدر
game_name = "Elden_Ring"
os.makedirs(f"dataset/{game_name}", exist_ok=True)

# مسار الفيديو اللي حملته
video = cv2.VideoCapture('elden_ring_gameplay.mp4')

count = 0
frame_rate = 30 # لو الفيديو 30 فريم في الثانية، كده هياخد صورة كل ثانية

while True:
    success, frame = video.read()
    if not success:
        break # لو الفيديو خلص اقفل
        
    if count % frame_rate == 0:
        # حفظ الصورة
        cv2.imwrite(f"dataset/{game_name}/img_{count}.jpg", frame)
        
    count += 1

print("تم استخراج الصور بنجاح!")