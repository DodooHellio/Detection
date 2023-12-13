import cv2

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_fullbody.xml")
upperbody_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_upperbody.xml")


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



cap = cv2.VideoCapture('./Assets/wake_video1.mp4')

while True:
    _, frame = cap.read()
    resized_frame = ResizeWithAspectRatio(frame, width=400)

    #desaturate the image
    gray = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
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


    cv2.imshow("Video", resized_frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyWindow()