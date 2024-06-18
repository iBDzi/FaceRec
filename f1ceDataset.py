import cv2
import os

cam = cv2.VideoCapture(0)
cam.set(3, 640) # длина размера изображения видеозахвата
cam.set(4, 480) # высота размера изображения видеозахвата

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Для каждого пользователя вписывается идентификатор
face_id = input('\n введите id пользователя и нажмите <return> ==>  ')

print("\n [Инфо] Инициализация захвата лица. Смотрите в камеру и ждите...")
# Инициализирует подсчет лиц для индивидуальной выборки
count = 0

while(True):

    ret, img = cam.read()
    img = cv2.flip(img, 1) # переворачивает видео по вертикали
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:

        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)     
        count += 1

        # Сохраняет захваченное изображение в папке датасета
        cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])

        cv2.imshow('image', img)

    k = cv2.waitKey(100) & 0xff # Нажатие 'ESC' для выхода из видеозахвата
    if k == 27:
        break
    elif count >= 30: # Делает 30 снимков лица и останавливает видеозахват
         break

# Небольшая очистка
print("\n [Инфо] Очистка и выход из программы..")
cam.release()
cv2.destroyAllWindows()


