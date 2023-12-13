import cv2
import time
import tensorflow as tf
from MultiPoseMovenetModule import MultiPoseMovenet


EDGES = {
    (0, 1): 'm',
    (0, 2): 'c',
    (1, 3): 'm',
    (2, 4): 'c',
    (0, 5): 'm',
    (0, 6): 'c',
    (5, 7): 'm',
    (7, 9): 'm',
    (6, 8): 'c',
    (8, 10): 'c',
    (5, 6): 'y',
    (5, 11): 'm',
    (6, 12): 'c',
    (11, 12): 'y',
    (11, 13): 'm',
    (13, 15): 'm',
    (12, 14): 'c',
    (14, 16): 'c'
}



if __name__ == '__main__':

    multipose = MultiPoseMovenet(confidence_threshold=0.7)

    cap = cv2.VideoCapture('Videos/1.mp4')
    while True:
        success, frame = cap.read()

        # Detection section
        results, keypoints_with_scores = multipose.find_postures(frame)
        print(frame.shape)
        print(keypoints_with_scores)


        # Render keypoints
        multipose.loop_through_people(frame=frame, keypoints_with_scores=keypoints_with_scores, edges=EDGES)

        cv2.imshow('Movenet Multipose', frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

    print(keypoints_with_scores)
