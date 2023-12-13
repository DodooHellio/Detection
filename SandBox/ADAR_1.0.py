import cv2
import time
import datetime

# Initialisation de la detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_fullbody.xml")


# Charger la vidéo
input_video_path = './Assets/P1077392.MP4'
cap = cv2.VideoCapture(input_video_path)

# Obtenir les informations de la vidéo (largeur, hauteur, etc.)
width = int(cap.get(3))
height = int(cap.get(4))

# Définir la résolution réduite
new_width = int(width * 0.5)  # Vous pouvez ajuster ce facteur selon vos besoins
new_height = int(height * 0.5)

# Définir le chemin pour la nouvelle vidéo
output_video_path = './Assets/GX010087_smallerrez.MP4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Format de compression vidéo
out = cv2.VideoWriter(output_video_path, fourcc, cap.get(5), (new_width, new_height))
frame_size = (new_width, new_height)


# Parametre pour programme de detection
detection = False
dectection_stopped_time = None
timer_started = False
timer_frames = 0
SECONDS_TO_RECORD_AFTER_DETECTION = 20
FRAMES_TO_RECORD_AFTER_DETECTION = 15
count = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Redimensionner le cadre
    small_frame = cv2.resize(frame, (new_width, new_height))

    # Desaturation de l'image
    gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)

    # dectectMultiScale(image, skill factor, accuracy)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    bodies = body_cascade.detectMultiScale(gray, 1.3, 5)


    # rectangle on the image for faces
    for (x, y, width, height) in faces:
        cv2.rectangle(small_frame, (x, y), (x + width, y + height), (255, 0, 0), 3)
    # rectangle on the image for bodies
    for (x, y, width, height) in bodies:
        cv2.rectangle(small_frame, (x, y), (x + width, y + height), (0, 255, 0), 3)


    # recording conditions
    # if any face or body in the frame
    # if len(faces) + len(bodies) > 0:
    if len(faces) > 0:
        timer_frames = 0
        if detection:
            timer_started = False
        else:
            detection = True
            current_time = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
            out = cv2.VideoWriter(f"{current_time}.mp4", fourcc, 20, frame_size)
            print("Started recording!")

    # if no face nor body in the frame and we are recording
    elif detection:
        if timer_started:
            #if time.time() - detection_stopped_time >= SECONDS_TO_RECORD_AFTER_DETECTION:
            if FRAMES_TO_RECORD_AFTER_DETECTION - timer_frames <= 0:
                detection = False
                timer_started = False
                out.release()
                print("Stop recording !")
            else :
                timer_frames += 1
                print(f"{timer_frames}/{FRAMES_TO_RECORD_AFTER_DETECTION}")

        else:
            timer_started = True
            print(f"Timer Started or reStarted")

            #detection_stopped_time = time.time()


    if detection:
        out.write(small_frame)

    #cv2.imshow("Camera", small_frame)
    count += 1
    print(f"frame number {count}")

    if cv2.waitKey(1) == ord('q'):
        break

# Release the ressources
out.release()
cap.release()
cv2.destroyWindow()
