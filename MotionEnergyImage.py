import os
import numpy as np
import cv2
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

# Motion energy image
def mei(video):
    cap = cv2.VideoCapture(video)
    ret, prev_frame = cap.read()
    if not ret:
        return None
    prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    m = np.zeros_like(prev_frame, dtype=float)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(frame, prev_frame)
        _, diff = cv2.threshold(diff, 25, 1, cv2.THRESH_BINARY)
        m = m + diff
        prev_frame = frame

    cap.release()
    m = cv2.normalize(m, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    return m

# Initializations
folder = r'C:\Users\Alex\Desktop\New folder'
files = ['Clic', 'No', 'Rotate', 'StopGraspOk']

training_seqs = []
training_seqlabels = []
test_seqs = []
test_seqlabels = []

# Process each file
for s in files:
    gesture_dir_path = os.path.join(folder, s)
    for seq in range(1, 15):  
        seq_path = os.path.join(gesture_dir_path, f'Seq{seq}')
        video_path = os.path.join(seq_path, 'output.avi')
        if os.path.exists(video_path):
            m = mei(video_path)
            m_flat = m.flatten()
            # Seq 1 to 5 for training
            if seq <= 5:
                training_seqs.append(m_flat)
                training_seqlabels.append(s)
            # Seq 6 to 15 for testing
            elif seq >= 6 and seq <= 15:
                test_seqs.append(m_flat)
                test_seqlabels.append(s)
# KNN 
k = 3
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(training_seqs, training_seqlabels)

predictions = knn.predict(test_seqs)
# Confusion matrix
matrix = confusion_matrix(test_seqlabels, predictions)
print(matrix)