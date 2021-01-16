import cv2
from keras.engine.saving import load_model
import sudoku_finder as s
import matplotlib.pyplot as plt
import numpy as np

IMAGE_SIZE = 256
model = load_model("sudoku_classification_model.h5")
cap = cv2.VideoCapture(0)

if not (cap.isOpened()):
    print("Camera could not be open !!!")

while True:
    ret, frame = cap.read()

    cv2.imshow("frame", frame)
    img = frame
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    img = np.array(img).reshape(-1,IMAGE_SIZE, IMAGE_SIZE, 1)
    p = model.predict([[img]])

    if s.detected(frame) and p[0][0] < p[0][1]:
        print("detected")
        img = s.sudoku_finder(frame)
        plt.imshow(img)
        plt.show()
        # cap.release()
        continue

    if cv2.waitKey(1) and 0xFF == ord('q'):
        cap.release()
        quit()
        cv2.destroyAllWindows()
        break
