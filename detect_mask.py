from keras.models import load_model
import cv2
import numpy as np

model = load_model('model-016.model')

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

labels_dict = {0:'NO MASK',1:'MASK'}
color_dict = {0:(0,0,255),1:(0,255,0)}
i = 0
while (True):
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    #if faces == []:
     #   cv2.putText(img, 'MASK')

    for x, y, w, h in faces:
        face_img = gray[y:y + w, x:x + w]
        resized = cv2.resize(face_img, (100, 100))
        normalized = resized / 255.0
        reshaped = np.reshape(normalized, (1, 100, 100, 1))
        result = model.predict(reshaped)

        label = np.argmax(result, axis=1)[0]
        if label < 0.5:
            crop_face = img[y:y + h, x:x + w]
            cv2.rectangle(img, (x, y), (x + w, y + h), color_dict[label], 2)
            cv2.imwrite(r'uploads\saved_img_{}.jpg'.format(i), crop_face)
            i = i+1
    cv2.imshow('LIVE', img)
    key = cv2.waitKey(10)

    if (key == 27):
        break

cv2.destroyAllWindows()
cap.release()