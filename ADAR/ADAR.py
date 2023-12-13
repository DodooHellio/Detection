'''
Automatic Detection And Recording program
https://www.youtube.com/watch?v=Exic9E5rNok
'''

import cv2
import time
import datetime


def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    return cv2.resize(image, dim, interpolation=inter)



#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('./Assets/wake_video2.mp4')

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_fullbody.xml")
upperbody_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_upperbody.xml")



detection = False
dectection_stopped_time = None
timer_started = False
SECONDS_TO_RECORD_AFTER_DETECTION = 5

#frame size and format for video export
frame_size = (int(cap.get(3)), int(cap.get(4)))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")


while True:
    _, frame = cap.read()

    #desaturate the image
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #dectectMultiScale(image, skill factor, accuracy)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    bodies = body_cascade.detectMultiScale(gray, 1.3, 5)
    upperbodies = upperbody_cascade.detectMultiScale(gray, 1.3, 5)

    #rectangle on the image for faces
    for (x, y, width, height) in faces:
        cv2.rectangle(frame, (x,y), (x + width, y + height), (255, 0, 0), 3)
    #rectangle on the image for bodies
    for (x, y, width, height) in bodies:
        cv2.rectangle(frame, (x,y), (x + width, y + height), (0, 255, 0), 3)
    # rectangle on the image for upperbodies
    for (x, y, width, height) in upperbodies:
        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 0, 255), 3)

    #recording conditions
    #if any face or body in the frame
    if len(faces) + len(bodies) > 0:
        if detection:
            timer_started = False
        else:
            detection = True
            current_time = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
            out = cv2.VideoWriter(f"{current_time}.mp4", fourcc, 20, frame_size)
            print("Started recording!")

    #if no face nor body in the frame and we are recording
    elif detection:
        if timer_started:
            if time.time() - detection_stopped_time >= SECONDS_TO_RECORD_AFTER_DETECTION:
                detection = False
                timer_started = False
                out.release()
                print("Stop recording !")
        else :
            timer_started = True
            detection_stopped_time = time.time()


    if detection:
        out.write(frame)

    #display the image on the screen but optional
    resized_frame = ResizeWithAspectRatio(frame, width=1920)

    cv2.imshow("Camera", resized_frame)

    if cv2.waitKey(1) == ord('q'):
        break

#Release the ressources
out.release()
cap.release()
cv2.destroyWindow()