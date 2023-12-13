import tensorflow as tf
import tensorflow_hub as hub
import cv2
from matplotlib import pyplot as plt
import numpy as np

""""
https://www.kaggle.com/models/google/movenet/frameworks/tensorFlow2/variations/multipose-lightning/versions/1
"""

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


class MultiPoseMovenet():
    def __init__(self, loadmodel="https://www.kaggle.com/models/google/movenet/frameworks/TensorFlow2/variations/multipose-lightning/versions/1", signatures='serving_default', confidence_threshold=0.5):

        # Optional if you are using a GPU
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        self.model = hub.load(loadmodel)
        self.movenet = self.model.signatures[signatures]
        self.confidence_threshold = confidence_threshold



    # Function to loop through each person detected and render

    def find_postures(self, frame):

        img = frame.copy()
        img = tf.image.resize_with_pad(tf.expand_dims(img, axis=0), 384, 640)
        input_img = tf.cast(img, dtype=tf.int32)
        # Detection section
        self.results = self.movenet(input_img)
        self.keypoints_with_scores = self.results['output_0'].numpy()[:, :, :51].reshape((6, 17, 3))

        return self.results, self.keypoints_with_scores

    def draw_keypoints(self, frame, keypoints):
        y, x, c = frame.shape
        shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))

        for kp in shaped:
            ky, kx, kp_conf = kp
            if kp_conf > self.confidence_threshold:
                cv2.circle(frame, (int(kx), int(ky)), 6, (0, 255, 0), -1)

    def draw_connections(self, frame, keypoints, edges):
        y, x, c = frame.shape
        shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))

        for edge, color in edges.items():
            p1, p2 = edge
            y1, x1, c1 = shaped[p1]
            y2, x2, c2 = shaped[p2]

            if (c1 > self.confidence_threshold) & (c2 > self.confidence_threshold):
                cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 4)

    def loop_through_people(self, frame, keypoints_with_scores, edges):
        for person in keypoints_with_scores:
            MultiPoseMovenet.draw_connections(frame, person, edges)
            MultiPoseMovenet.draw_keypoints(frame, person)





