from Required_Functions import *

from Solver import *


class main:
    def __init__(self):
        self.cap=cv2.VideoCapture(0)                # Creating an object for video capturing

            
    def detected(self):


        while True:                                                # capturing frame-by-frame                  
            success, img = self.cap.read()                          # cap.read() returns a boolean value
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
                    image2=ans(bg, mask, pts1, pts2, img)
                    rect,qe_image=cv2.imencode('.jpg', image2)
                    if cv2.waitKey(1) & 0xff == ord('q'):
                        break
                    return qe_image.tobytes()
                    # cv2.imshow("tt",image2)
                
    def __del__(self):
        self.cap.release()


# m = main()
# print(m.detected())

# 0 0 0 0 0 0 0 0 2
# 0 0 0 0 0 0 9 4 0
# 0 0 3 0 0 0 0 0 5
# 0 9 2 3 0 5 0 7 4
# 8 4 0 0 0 0 0 0 0
# 0 6 7 0 9 8 0 0 0
# 0 0 0 7 0 6 0 0 0
# 0 0 0 9 0 0 0 2 0
# 4 0 8 5 0 0 3 6 0