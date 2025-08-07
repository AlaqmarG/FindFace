import cv2
import numpy as np

face_classifier = cv2.CascadeClassifier('frontal_face.xml')
eye_classifier = cv2.CascadeClassifier('eye.xml')

def detect_faces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    eyes = eye_classifier.detectMultiScale(gray, 1.3, 5)

    # Return early if there are no faces
    if faces is ():
        return image
    
    # Draw a rectangle around any faces detected
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

    for (x, y, w, h) in eyes:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

    return image

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame = detect_faces(frame)

    cv2.imshow("Face Finder", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()