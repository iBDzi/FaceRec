import cv2
import numpy as np
from PIL import Image
import os

# Path for face image database
path = 'dataset'

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml");

# function to get the images and label data
def getImagesAndLabels(path):

    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]     
    faceSamples=[]
    ids = []

    for imagePath in imagePaths:

        PIL_img = Image.open(imagePath).convert('L') # преобразование в серый оттенок
        img_numpy = np.array(PIL_img,'uint8')

        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = detector.detectMultiScale(img_numpy)

        for (x,y,w,h) in faces:
            faceSamples.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)

    return faceSamples,ids

print ("\n [Инфо] Изучение лиц...")
faces,ids = getImagesAndLabels(path)
recognizer.train(faces, np.array(ids))

# Сохраняет модель в trainer/trainer.yml
recognizer.write('trainer/trainer.yml')

# Выводит количество изученных лиц и заканчивает программу
print("\n [Инфо] {0} лиц изучено. Выход из программы".format(len(np.unique(ids))))
