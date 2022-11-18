import cv2
import imshow
import face_recognition

from google.colab.patches import cv2_imshow

img = cv2.imread('/content/zayn malick.jfif', cv2.IMREAD_UNCHANGED)
rgb_img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
img_encoding = face_recognition.face_encodings(rgb_img)[0]
cv2_imshow(img)