# -*- coding: utf-8 -*-
import numpy as np
import os
import re
from sklearn.svm import SVC
from sklearn.metrics import precision_recall_curve, recall_score, precision_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools

# train_path = 'D:/workspace/github/garbage_dumping_detection/additional/train'
# test_path = 'D:/workspace/github/garbage_dumping_detection/additional/test'

train_path = 'C:/Users/JM/Desktop/Data/ETRIrelated/additional/3_train'
test_path = 'C:/Users/JM/Desktop/Data/ETRIrelated/additional/3_test'


def data_load(_data_path):
    data = []
    label = []
    split_label = []
    for file in os.listdir(_data_path):
        file_path = os.path.join(_data_path, file)
        f = open(file_path, 'r')

        for line in f.readlines():
            # split_line = line.split(' ')
            split_line = re.split('[, ]+', line)
            a = int(split_line[-1])
            split_label.append(a)

            if a == 0:
                label.append(0)
            else:
                label.append(1)

            split_line = split_line[3:-1]
            dat = []
            for idx, x in enumerate(split_line):
                if idx % 3 == 2:
                    continue

                dat.append(float(x))

            dat = normalize_pose_(dat)
            dat = scaling_data_(dat)
            data.append(dat)

    return np.asarray(data), np.asarray(label), np.asarray(split_label)


def normalize_pose_(_pose_data):

    neck_x = _pose_data[2]
    neck_y = _pose_data[3]
    # base_index = 0
    for base_index in range(18):
        _pose_data[base_index * 2] -= neck_x
        _pose_data[base_index * 2 + 1] -= neck_y  # 목좌표로 좌표계 변환

    return _pose_data


########################################################################
#           scaling the data using neck to shoulder distance           #
########################################################################
def scaling_data_(_pose_data):

    neck = [_pose_data[2], _pose_data[3]]
    right_factor = [_pose_data[4], _pose_data[5]]
    left_factor = [_pose_data[10], _pose_data[11]]
    # right_factor = [_pose_data[16], _pose_data[17]]
    # left_factor = [_pose_data[22], _pose_data[23]]

    # left_dist = ((left_shoulder[0] - neck[0]) ** 2 + (left_shoulder[1] - neck[1]) ** 2) ** 0.5
    # right_dist = ((right_shoulder[0] - neck[0]) ** 2 + (right_shoulder[1] - neck[1]) ** 2) ** 0.5

    left_dist = ((left_factor[0] - neck[0]) ** 2 + (left_factor[1] - neck[1]) ** 2) ** 0.5
    right_dist = ((right_factor[0] - neck[0]) ** 2 + (right_factor[1] - neck[1]) ** 2) ** 0.5

    # dist = max(left_dist, right_dist)
    dist = (left_dist + right_dist) / 2
    base_index = 0

    while base_index < 18:
        _pose_data[base_index * 2] /= dist
        _pose_data[base_index * 2 + 1] /= dist
        base_index += 1

    return _pose_data


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    plt.show()


def load_npy(_data_path):

    data = []
    label = []
    for file in os.listdir(_data_path):
        file_path = os.path.join(_data_path, file)
        dat = np.load(file_path)
        dat = dat.reshape((840))
        data.append(dat)
        split_file_name = re.split('[-. ]+', file)
        label.append(int(split_file_name[-2]))

    return np.asarray(data), np.asarray(label)

if __name__ == '__main__':

    # X_train, y_train, y_split_train = data_load(train_path)
    # X_test, y_test, y_split_test = data_load(test_path)

    """
    X_train, y_train = load_npy(train_path)
    print(X_train.shape)
    X_test, y_test = load_npy(test_path)
    svm = SVC(class_weight={0:0.06, 1:0.94})
    svm.fit(X_train, y_train)
    # pred = svm.predict(X_test)

    y_score = svm.score(X_test, y_test)
    y_score = svm.decision_function(X_test)
    y_pred = svm.predict(X_test)

    print(svm.score(X_test, y_test))
    print(recall_score(y_test, y_pred))
    print(precision_score(y_test, y_pred))
    # print(precision_recall_fscore_support(y_test, y_pred,average='macro'))

    final = []
    for score in y_score:

        if score < -1.05:
            final.append(0)
        else:
            final.append(1)

    final = np.asarray(final)
    print(recall_score(y_test, final))
    print(precision_score(y_test, final))
    """

    train_path_1 = 'C:/Users/JM/Desktop/Data/ETRIrelated/additional/1_train'
    test_path_1 = 'C:/Users/JM/Desktop/Data/ETRIrelated/additional/1_test'
    train_path_2 = 'C:/Users/JM/Desktop/Data/ETRIrelated/additional/2_train'
    test_path_2 = 'C:/Users/JM/Desktop/Data/ETRIrelated/additional/2_test'
    train_path_3 = 'C:/Users/JM/Desktop/Data/ETRIrelated/additional/3_train'
    test_path_3 = 'C:/Users/JM/Desktop/Data/ETRIrelated/additional/3_test'

    X1_train, y1_train = load_npy(train_path_1)
    X2_train, y2_train = load_npy(train_path_2)
    X3_train, y3_train = load_npy(train_path_3)

    X1_test, y1_test = load_npy(test_path_1)
    X2_test, y2_test = load_npy(test_path_2)
    X3_test, y3_test = load_npy(test_path_3)

    X1 = X1_train.tolist()
    X2 = X2_train.tolist()
    X3 = X3_train.tolist()

    print(len(X1))
    X1.extend(X2)
    X1.extend(X3)

    y1 = y1_train.tolist()
    a = len(y1)
    y2 = y2_train.tolist()
    b = len(y2)
    y3 = y3_train.tolist()
    c = len(y3)

    y1.extend(y2)
    y1.extend(y3)

    svm = SVC(C=0.1,
              kernel='rbf')
    svm.fit(X1, y1)

    Y_score = svm.decision_function(X1)
    print(len(Y_score))
    final = []
    for score in Y_score:

        if score < -1.2:
            final.append(0)
        else:
            final.append(1)

    final = np.asarray(final)
    print(recall_score(y1, final))
    print(precision_score(y1, final))

    arridx = np.where(final == 1)
    for idx, i in enumerate(y1):
        if idx < a:
            y1[idx] *= 1

        elif a < idx < a + b:
            y1[idx] *= 2

        else:
            y1[idx] *= 3

    X1 = np.asarray(X1)[arridx[0]]
    Y1 = np.asarray(y1)[arridx[0]]

    # print(np.where(Y1 == 2))

    svm2 = SVC(C=1,
               class_weight={0: 0.1,
                             1: 0.4,
                             2: 0.2,
                             3: 0.2})
    svm2.fit(X1, Y1)

    X1_test = X1_test.tolist()
    X2_test = X2_test.tolist()
    X3_test = X3_test.tolist()
    X_test = X1_test
    X_test.extend(X2_test)
    X_test.extend(X3_test)

    y1_test = (y1_test*1).tolist()
    y2_test = (y2_test*2).tolist()
    y3_test = (y3_test*3).tolist()

    Y_test = y1_test
    Y_test.extend(y2_test)
    Y_test.extend(y3_test)

    Y_pred = svm2.predict(X_test)
    cnf = confusion_matrix(Y_test, Y_pred)
    print(cnf)
    class_names = ['walk', 'bending', 'Drop', 'Throw']
    plot_confusion_matrix(cnf, classes=class_names, normalize=True,
                          title='Confusion matrix, without normalization')
