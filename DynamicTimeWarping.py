import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, accuracy_score

# Dynamic Time Warping
def DTW(s, t, window_size):
    ns = s.shape[0]
    nt = t.shape[0]
    window_size = max(window_size, abs(ns-nt))

    D = np.zeros((ns+1, nt+1))
    D += np.inf
    D[0, 0] = 0

    for i in range(ns):
        for j in range(max(i - window_size, 0), min(i + window_size, nt)):
            cost = euclidean(s[i], t[j])
            D[i+1, j+1] = cost + min(D[i, j+1], D[i+1, j], D[i, j])

    return D[ns, nt]

# Initializations
folder = r'C:\Users\Alex\Desktop\New folder'
files = ['Clic', 'No', 'Rotate', 'StopGraspOk']
gestures = {gesture: idx for idx, gesture in enumerate(files)}
k = 2
window_size = 50

scaler = MinMaxScaler()
training_seqs = []
training_seqlabels = []
test_seqs = []
test_seqlabels = []

# Process each file
for g in files:
    seqs = [d for d in os.listdir(os.path.join(folder, g)) if 'Seq' in d]
    
    for idx, seq in enumerate(seqs):
        vid = os.path.join(folder, g, seq, 'output.avi')
        readvid = cv2.VideoCapture(vid)

        frames = []
        while readvid.isOpened():
            ret, frame = readvid.read()
            if not ret:
                break
            # Convert to grayscale and binary
            gframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            bframe = cv2.threshold(gframe, 128, 255, cv2.THRESH_BINARY)[1]
            
            moments = cv2.moments(bframe)
            cx = moments['m10'] / moments['m00']
            cy = moments['m01'] / moments['m00']
            frames.append([cx, cy])
        
        features = np.array(frames)
        # Seq 1 to 5 for training
        if idx < 5:
            training_seqs.append(scaler.fit_transform(features))
            training_seqlabels.append(gestures[g])
        # Seq 6 to 15 for testing
        else:
            test_seqs.append(scaler.transform(features))
            test_seqlabels.append(gestures[g])

# DTW distances
dtwdistances = np.zeros((len(training_seqs), len(test_seqs)))
for i in range(len(training_seqs)):
    for j in range(len(test_seqs)):
        dtwdistances[i, j] = DTW(training_seqs[i], test_seqs[j], window_size)

# KNN
predictions = []
for j in range(len(test_seqs)):
    nearn = np.argsort(dtwdistances[:, j])[:k]
    c = np.bincount([training_seqlabels[nn] for nn in nearn])
    predictions.append(np.argmax(c))

# Confusion matrix
conf_matrix = confusion_matrix(test_seqlabels, predictions)

# Compute accuracy
accuracy = accuracy_score(test_seqlabels, predictions)

print(f'Predictions accuracy: {accuracy*100}%')

# Plots
plt.figure(figsize=(10,7))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Greens) 
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(files))
plt.xticks(tick_marks, files, rotation=45)
plt.yticks(tick_marks, files)

thresh = conf_matrix.max() / 2.
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):    plt.text(j, i, format(conf_matrix[i, j], 'd'),
             horizontalalignment="center",
             color="white" if conf_matrix[i, j] > thresh else "black")

plt.tight_layout()
plt.show()