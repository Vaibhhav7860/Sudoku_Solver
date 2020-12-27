
from Required_Functions import *
import sys
from Solver import *

sys.setrecursionlimit(100000)
class main:
    def __init__(self):
        self.cap=cv2.VideoCapture(0)                # Creating an object for video capturing

    def show_output(self):

        while True:
            success, img = self.cap.read()             # capturing frame-by-frame
                                               # cap.read() returns a boolean value
                                               # True ---> if image frame is read correctly
                                               # False ---> if image frame is read incorrectly
            img = np.asarray(img)
            warp, pts1, pts2, flag = warped(img, 0)
            if flag == True:
                thresh = extract(warp,0)
                arr, av = getmat(thresh)
                sol = solve_sudoku(arr)
                #print (sol)
                if sol:
                    bg, mask = output(warp,return_sudoku(matrix),av,arr)
                    ans(bg, mask, pts1, pts2, img)

            cv2.imshow("Img", img)
            if cv2.waitKey(1) & 0xff == ord('q'):
                break
    def __del__(self):
        self.cap.release()


m = main()
m.show_output()
