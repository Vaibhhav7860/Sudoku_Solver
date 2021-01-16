import cv2
import operator
import numpy as npy
from matplotlib import pyplot as plt


# find distance between 2 points
def distance_between(p1, p2):
    a = p2[0] - p1[0]
    b = p2[1] - p1[1]
    return npy.sqrt((a ** 2) + (b ** 2))


def detected(img):
    m = img
    #original_image = m
    # ----------------------------------------------------------------------
    # pre process image
    original_image = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)

    image = cv2.GaussianBlur(original_image.copy(), (9, 9), 0)
    image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    image = cv2.bitwise_not(image, image)
    kernel = npy.array([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]], npy.uint8)
    image = cv2.dilate(image, kernel)

    # plt.show()

    # find puzzle
    contours, h = cv2.findContours(image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    p = contours[0]
    # print("------------------------------------------------------------------")
    # print(p)
    # print(cv2.contourArea(p))
    # we cannot detect images in which puzzle is very small otherwise it will be blur
    if cv2.contourArea(p) > 9000:
        # print("detected")
        # print("###################################################################################")
        return 1
    return 0


def sudoku_finder(img):
    m = img
    #original_image = m
    #plt.imshow(m)
    #plt.show()

    # ----------------------------------------------------------------------
    # pre process image
    original_image = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)

    image = cv2.GaussianBlur(original_image.copy(), (9, 9), -3)
    image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    image = cv2.bitwise_not(image, image)
    kernel = npy.array([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]], npy.uint8)
    image = cv2.dilate(image, kernel)

    #plt.imshow(image)
    #plt.show()

    # find puzzle
    contours, h = cv2.findContours(image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    p = contours[0]
    # print("------------------------------------------------------------------")
    # print(p)
    # print(cv2.contourArea(p))

   # cv2.drawContours(m, p, -1, (255, 0, 0), thickness=6)

    #plt.imshow(m)
    #plt.show()

    bottom_right, _ = max(enumerate([pt[0][0] + pt[0][1] for pt in p]), key=operator.itemgetter(1))
    top_left, _ = min(enumerate([pt[0][0] + pt[0][1] for pt in p]), key=operator.itemgetter(1))
    bottom_left, _ = min(enumerate([pt[0][0] - pt[0][1] for pt in p]), key=operator.itemgetter(1))
    top_right, _ = max(enumerate([pt[0][0] - pt[0][1] for pt in p]), key=operator.itemgetter(1))
    
    corners = [p[top_left][0], p[top_right][0], p[bottom_right][0], p[bottom_left][0]]
    # croping the image

    top_left, top_right, bottom_right, bottom_left = corners[0], corners[1], corners[2], corners[3]
    
    src = npy.array([top_left, top_right, bottom_right, bottom_left], dtype='float32')
    # find the side with max distance as it will be needed when wrapping the image i.e convert 3d plane to 2d plane
    side = max([distance_between(bottom_right, top_right), distance_between(top_left, bottom_left), distance_between(bottom_right, bottom_left), distance_between(top_left, top_right)])
    side = side-1
    print("side = ", side)

    dst = npy.array([[0, 0], [side, 0], [side, side], [0, side]], dtype='float32')
    m = cv2.getPerspectiveTransform(src, dst)
    image = cv2.warpPerspective(original_image, m, (int(side), int(side)))

    #plt.imshow(image)
    #plt.show()

    return image, side



#path = "C:\Users\ACER\Desktop\Sudoku1.png"
#img = cv2.imread(path)
#sudoku_finder(img)
