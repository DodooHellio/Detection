import cv2
import mediapipe as mp
import time
#from FaceDetectionModule import *
from PoseModule import poseDetector


def zoom_at(img, zoom, coord=None):
    """
    Function to zoom on the ROI (region of interest)
    Centered at "coord", if given, else the image center.


    :param img: numpy.ndarray of shape (h,w,:)
    :param zoom: int
    :param coord: int
    :return:
    """

    # Translate to zoomed coordinates
    h, w, _ = [zoom * i for i in img.shape]
    if coord is None:
        cx, cy = w / 2, h / 2
    else:
        cx, cy = [zoom * c for c in coord]

    #img = cv2.resize(img, (0, 0), fx=zoom, fy=zoom)
    """y1 = int(round(cy - h / zoom * .5))
    y2 = int(round(cy + h / zoom * .5))
    x1 = int(round(cx - w / zoom * .5))
    x2 = int(round(cx + w / zoom * .5))"""

    y1 = int(round(cy - 360))
    if y1 <0:
        y1 = 0
    y2 = int(round(y1 + 920))
    x1 = int(round(cx - 360))
    if x1 <0:
        x1 = 0
    x2 = int(round(x1 + 720))

    img = img[y1:y2, x1: x2, :]

    print(f'zoomed high h = {h}, zoomed width w = {w}, cx = {cx}, cy = {cy}')
    print(f" x1 = {x1}, x2 = {x2}, y1 = {y1}, y2 = {y2}")
    return img


def crop_on_subject(img, cx: int, cy: int, h_rez = 1280, w_rez = 720):
    """
    Function for cropping an image around a point with x, y coordinates and resolution
    :param img: numpy.ndarray of shape (h,w,c)
    :param x: int
    :param y: int
    :return: img
    """
    cx = cx
    cy = cy
    cx_offset = 360
    cy_offset = 360


    y1 = int(round(cy-cy_offset))
    if y1 <0:
        y1 = 0
    y2 = int(round(y1 + h_rez))
    x1 = int(round(cx - cx_offset))
    if x1 <0:
        x1 = 0
    x2 = int(round(x1 + w_rez))

    img = img[y1:y2, x1: x2, :]
    return img


# Function to merge the resized ROI (region of interest) back into the original frame
def merge_roi(img, roi, x, y):
    """
    Function to merge the resized ROI (region of interest) back into the original frame at the specific x, y coord
    :param img: numpy.ndarray of shape (h,w,c)
    :param roi: numpy.ndarray of shape (h,w,c)
    :param x: int
    :param y: int
    :return: img
    """
    h, w, _ = roi.shape
    img[y:y+h, x:x+w, :] = roi
    return img





if __name__ == '__main__':

    cap = cv2.VideoCapture("Videos/3.mp4")

    width = int(cap.get(3))
    height = int(cap.get(4))
    new_width = int(width * 0.25)
    new_height = int(height * 0.25)

    pTime = 0
    detector = poseDetector()

    while True:
        success, img = cap.read()
        #img = cv2.resize(img, (new_width, new_height))
        results = detector.pose.process(img)

        print(results.pose_landmarks)
        img = detector.findPose(img)
        lmList = detector.findPosition(img, draw=False)
        if len(lmList) != 0:
            print(lmList[14])
            #cv2.circle(img, (lmList[0][1], lmList[0][2]), 25, (255, 0, 255))
            cv2.rectangle(img, (lmList[0][1]-50,lmList[0][2]-50), (lmList[0][1]+50,lmList[0][2]+50), (255, 0, 255), 1)

            x = lmList[0][1]
            y = lmList[0][2]
            #print(f"Coord noze : {x,y}")
            #roi = zoom_at(img, 1, coord=(x, y))
            roi = crop_on_subject(img, x, y)
            small_roi = cv2.resize(roi, (0, 0), fx=0.5, fy=0.5)  # Réduction à la moitié de la résolution

            try :
                img = merge_roi(img, small_roi, 0,0)
            except ValueError:
                print("merge_roi doesn't work")
                pass


        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)
        cv2.imshow("Image", img)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyWindow("Image")