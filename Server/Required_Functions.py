import numpy as np
import cv2
from imutils import grab_contours
from skimage.segmentation import clear_border
from keras.models import load_model

def getpoints(t):
    pts = t.reshape((4, 2))
    q = np.zeros(4)
    for i in range(4):
        q[i] = pts[i][0] + pts[i][1]
    br = pts[np.argmax(q)]
    tl = pts[np.argmin(q)]
    pts = np.delete(pts, np.argmax(q), axis=0)
    q = np.delete(q, np.argmax(q))
    pts = np.delete(pts, np.argmin(q), axis=0)
    bl = pts[0]
    tr = pts[1]
    if bl[0] < tr[0]:
        temp = bl
        bl = tr
        tr = temp
    hratio = float((br[1] - tr[1]) / (bl[1] - tl[1]))
    wratio = float((br[0] - bl[0]) / (tr[0] - tl[0]))
    flag = True
    if hratio > 1.2 or hratio < 0.85 or wratio > 1.2 or wratio < 0.85:
        flag = False
    point = np.array([tl, tr, bl, br]).astype("float32")
    return point, flag

def getcontour(img):
    cont = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE, )
    cont = grab_contours(cont)
    cont = sorted(cont, key=cv2.contourArea, reverse=True)[:10]
    t = None
    flag = False
    for c in cont:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, peri * 0.1, True)
        if len(approx) == 4 and cv2.contourArea(c) > 10000:
            t = approx
            flag = True
            break
    return t, flag

def warped(img, debug):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 3)
    canny = cv2.Canny(blur, 40, 70)
    dil = cv2.dilate(canny, np.ones((5, 5)), iterations=2)
    erode = cv2.erode(dil, np.ones((3, 3)))
    pts2 = np.float32([[0, 0], [0, 460], [460, 0], [460, 460]])
    t, flag = getcontour(erode)
    flag1 = False
    flag2 = False
    if flag == True:
        pts1, flag2 = getpoints(t)
        mat = cv2.getPerspectiveTransform(pts1, pts2)
        warp = cv2.warpPerspective(img, mat, (460, 460))
        flag1 = True

    if debug and flag1:
        cv2.imshow("warped", warp)

    if flag1 or flag2:
        return warp, pts1, pts2, flag1 or flag2
    else:
        return img, pts2, pts2, flag1 or flag2

def extract(img, debug):
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    # cv2.cvtColor method is used to convert an image from one color space to another.
                                                    # cv2.COLOR_BGR2GRAY is a flag used for BGR->Gray conversion
    # cv2.threshold function assigns one value(maybe white) if pixel value is greater than threshold value
    # else assigns another value(maybe black)

    # cv2.threshold()
    # 1st argument --> The grayscaled source image.
    # 2nd argument --> Threshold value used to classify the pixel values
    # 3rd argument --> Third argument is the maxVal which represents the value
    # to be given if pixel value is more than (sometimes less than) the threshold value
    # 4th argument --> Type of thresholding to be done.
    # cv2.THRESH_OTSU is passed as an extra flag
    thresh = cv2.threshold(grey, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    thresh = clear_border(thresh)    # For removing the image borders
    # now thresh variable has the thresholded image.
    # For debugging 0 is passed so just the thresholded image will be returned.
    if debug:
        cv2.imshow("thresh", thresh)
    return thresh

def process(img):
    # In skimage, images are simply numpy arrays, which support a variety of data types 1, i.e. “dtypes”.
    # uint8 has a range from 0 to 255.
    # astype() is used for converting the type of the image.
    imggray = img.astype(np.uint8)
    canny = cv2.Canny(imggray, 50, 200)
    dilimg = cv2.dilate(canny, np.ones((3, 3), dtype="uint8"), iterations=1)
    erimg = cv2.erode(dilimg, np.ones((3, 3)))
    cont, h = cv2.findContours(erimg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours_sizes = [(cv2.contourArea(cnt), cnt) for cnt in cont]
    if (len(contours_sizes) == 0):
        return np.ones((40, 40))
    else:
        t = max(contours_sizes, key=lambda x: x[0])[1]
        x, y, w, h = cv2.boundingRect(t)
        fimg = img[y:y + h, x:x + w]
        fimg = cv2.copyMakeBorder(fimg, y, 40 - y - h, x, 40 - x - w, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        return fimg

def getmat(img):
    model = load_model("t1.h5")
    solve = np.zeros(shape=(9, 9))
    av = np.zeros((9, 9))
    for i in range(9):
        for j in range(9):
            aa = img[5 + (50 * i):5 + (50 * (i + 1)), 5 + (50 * j):5 + (50 * (j + 1))]
            aa = aa[5:45, 5:45]
            percentFilled = cv2.countNonZero(aa) / float(40 * 40)
            if percentFilled > 0.05:
                aa = process(aa)
                aa = 255 - aa
                aa = aa / 255.0
                aa = cv2.resize(aa, (28, 28))
                aa = np.reshape(aa, (1, 28, 28, 1))
                res = model.predict(aa)[0]
                val = np.argmax(res)
                ff = max(res)
                # print(val, ff)
                solve[i][j] = val
                av[i][j] = 1
            else:
                solve[i][j] = 0

    # print(solve)
    return solve, av

def generate_square_coords():
    square_coordinates = []
    row_begin = 0
    row_end = 2
    col_begin = 0
    col_end = 2

    while len(square_coordinates) < 9:
        square_coordinates.append({
            "row_begin": row_begin,
            "row_end": row_end,
            "col_begin": col_begin,
            "col_end": col_end
        })
        if col_begin < 6 and col_end < 8:
            col_begin += 3
            col_end += 3
        else:
            col_begin = 0
            col_end = 2
            row_begin += 3
            row_end += 3

    return tuple(square_coordinates)


def insert_sorted(zone, zones):
    if len(zones) == 0:
        zones.append(zone)
    else:
        inserted = False
        for i in range(0, len(zones)):
            if zone["len"] > zones[i]["len"]:
                zones.insert(i, zone)
                inserted = True
                break
            else:
                continue
        if not inserted:  # it was smaller than all of them
            zones.append(zone)



def extract_zones(board):
    zones = []
    for row in range(0, len(board)):
        zone = {}
        zone["type"] = "row"
        zone["len"] = np.count_nonzero(board[row])
        zone["coord"] = row
        insert_sorted(zone, zones)
    for col in range(0, 9):
        nr_elements = 0
        for row in range(0, 9):
            if board[row][col] != 0:
                nr_elements += 1
        zone = {}
        zone["type"] = "col"
        zone["len"] = nr_elements
        zone["coord"] = col
        insert_sorted(zone, zones)
    for square in generate_square_coords():
        nr_elements = 0
        for row in range(square["row_begin"], square["row_end"] + 1):
            for col in range(square["col_begin"], square["col_end"] + 1):
                if board[row][col] != 0:
                    nr_elements += 1
        zone = {}
        zone["type"] = "square"
        zone["len"] = nr_elements
        zone["coord"] = tuple(square.values())
        insert_sorted(zone, zones)

    return zones

def get_zone_elements(zone_type, coord1, coord2, board):
    elements = []
    if zone_type == "col":
        for row in range(0, 9):
            if board[row][coord2] != 0:
                elements.append(board[row][coord2])
    elif zone_type == "row":
        for col in range(0, 9):
            if board[coord1][col] != 0:
                elements.append(board[coord1][col])
    else:
        square_coords = generate_square_coords()
        for square in square_coords:
            if (square["row_begin"] <= coord1 <= square["row_end"]) and (
                    square["col_begin"] <= coord2 <= square["col_end"]):
                for row in range(square["row_begin"], square["row_end"] + 1):
                    for col in range(square["col_begin"], square["col_end"] + 1):
                        if board[row][col] != 0:
                            elements.append(board[row][col])
                break

    return elements

def insert_possibilities(puzzle, row, col):
    if puzzle[row][col] == 0:
        row_elements = get_zone_elements("row", row, col, puzzle)
        col_elements = get_zone_elements("col", row, col, puzzle)
        square_elements = get_zone_elements("square", row, col, puzzle)
        numbers = [number for number in range(1, 10)]
        possibilities = [i for i in range(1, 10)]
        for possibility in numbers:
            if (possibility in row_elements) or (possibility in col_elements) or (possibility in square_elements):
                possibilities.remove(possibility)
        if len(possibilities) == 1:
            puzzle[row][col] = possibilities[0]

def ans(bg, mask, pts1, pts2, originalimage):
    mat = cv2.getPerspectiveTransform(pts2, pts1)
    warp1 = cv2.warpPerspective(bg, mat, (originalimage.shape[1], originalimage.shape[0]))
    mask1 = cv2.warpPerspective(mask, mat, (originalimage.shape[1], originalimage.shape[0]))
    maskinv = cv2.bitwise_not(mask1)
    qq = cv2.bitwise_and(originalimage, originalimage, mask=maskinv)
    qw = cv2.bitwise_and(warp1, warp1, mask=mask1)
    qe = cv2.add(qq, qw)
    # cv2.imshow("tt", qe)
    # rect,qe_image=cv2.imencode('.jpg', qe)
    return qe

def output(img, sol, av,arr):
    overlay = img.copy()
    out = img.copy()
    bg = img.copy()
    bg[:] = [0, 0, 0]
    for i in range(9):
        for j in range(9):
            if av[j][i] == 0:
                cv2.putText(bg, str(int(sol[j][i])), (50 * i + 8, 50 + 50 * j - 2), cv2.FONT_HERSHEY_SIMPLEX, 1.5,(0, 0, 255), 3)
            else:
                cv2.putText(bg, str(int(arr[j][i])), (50 * i + 8, 50 + 50 * j - 2), cv2.FONT_HERSHEY_SIMPLEX, 1.5,(255, 0, 255), 3)

    mask = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
    r, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # mask=cv2.bitwise_not(mask)
    #cv2.imshow("bg", bg)
    return bg, mask


'''
def solver(puzzle):
    for itrations in range(0, 3):
        zones = extract_zones(puzzle)
        for zone in zones:
            if zone["type"] == "row":
                row = zone["coord"]
                for col in range(0, 9):
                    insert_possibilities(puzzle, row, col)
            elif zone["type"] == "col":
                col = zone["coord"]
                for row in range(0, 9):
                    insert_possibilities(puzzle, row, col)
            else:
                row_begin = zone["coord"][0]
                row_end = zone["coord"][1]
                col_begin = zone["coord"][2]
                col_end = zone["coord"][3]
                for row in range(row_begin, row_end + 1):
                    for col in range(col_begin, col_end + 1):
                        insert_possibilities(puzzle, row, col)
    return puzzle
'''