from Solver import *
from Required_Functions import *
import sys
#from Solver import *
import sudoku_finder as s
from matplotlib import pyplot as plt
sys.setrecursionlimit(100000)
class main:
    def __init__(self,IMAGE_SIZE=256):
        self.cap=cv2.VideoCapture(0)                # Creating an object for video capturing
        self.IMAGE_SIZE = IMAGE_SIZE
    def show_output(self):
        model = load_model("sudoku_classification_model.h5")

        while True:
            success, frame = self.cap.read()             # capturing frame-by-frame
                                               # cap.read() returns a boolean value
                                               # True ---> if image frame is read correctly
                                               # False ---> if image frame is read incorrectly
            #img = np.asarray(img)
            plt.imshow(frame)
            img = frame
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (self.IMAGE_SIZE, self.IMAGE_SIZE))
            img = np.array(img).reshape(-1, self.IMAGE_SIZE,self.IMAGE_SIZE,1)


            p = model.predict([[img]])

            if s.detected(frame) and p[0][0] < p[0][1]:
                print("detected")
                img, side = s.sudoku_finder(frame)
                plt.imshow(img)
                plt.show()
                # cap.release()
                flag = True
                if flag == True:
                    warp, pts1, pts2, flag = warped(img, 0)
                    thresh = extract(img,1)
                    #plt.imshow(thresh)
                    #plt.show()
                    arr, av = getmat(thresh, side)

                    matrix = inp(arr)
                    sol = solve_sudoku(matrix)
                    print(sol)
                    if sol:
                        bg, mask = output(img,return_sudoku(matrix),av,arr, side)
                        #ans(bg, mask, pts1, pts2, img)

                cv2.imshow("Img", bg)
                break
            if cv2.waitKey(1) & 0xff == ord('q'):
                break
    def __del__(self):
        self.cap.release()


m = main(256)
m.show_output()
