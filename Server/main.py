from flask import Flask, render_template, Response
from camera import main
import cv2
import sys

sys.setrecursionlimit(100000)

app = Flask(__name__ )

@app.route('/')
def index():
    return render_template('index.html')


def gen(camera):
    while True:
        frame1 = camera.stream()
        # image=camera.show_output()
        # ret,frame=cv2.imencode('.jpg', image)

        yield (b'--frame1\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + bytearray(frame1)+ b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(main()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def gen2(cam2):
    while True:
        frame = cam2.detected()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + bytearray(frame)+ b'\r\n\r\n')

@app.route('/detect')
def detect():
    return Response(gen2(main()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)
#     cap=cv2.VideoCapture(0)                # Creating an object for video capturing
#     while True:
#         success, img = cap.read()             # capturing frame-by-frame
#                                            # cap.read() returns a boolean value
#                                            # True ---> if image frame is read correctly
#                                            # False ---> if image frame is read incorrectly
#         img = np.asarray(img)
#         warp, pts1, pts2, flag = warped(img, 0)
#         if flag == True:
#             thresh = extract(warp,0)
#             arr, av = getmat(thresh)
#             sol = solve_sudoku(arr)
#             #print (sol)
#             if sol:
#                 bg, mask = output(warp,return_sudoku(matrix),av,arr)
#                 ans(bg, mask, pts1, pts2, img)

#         cv2.imshow("Img", img)
#         if cv2.waitKey(1) & 0xff == ord('q'):
#             break
# print('end')