import json
import os
from xml.etree.ElementTree import parse
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_recall_fscore_support
import numpy as np


files = [5, 15, 17, 18, 28,
         31, 41, 42, 58, 100,
         113, 115, 117, 120, 124,
         125, 127, 133, 136, 140,
         144, 147,  160, 161,
         164, 165, 172, 186, 191,
         196, 199, 202, 206, 213
         ]
# 153


def read_pose_(filename):
    f = open(filename, 'r')
    js = json.loads(f.read())
    f.close()

    return js


def check_macro_file(_file_num, _frame_num):
    macro_file_path = "C:\\Users\\JM\\Desktop\\Data\\ETRIrelated\\pose classification\\class1macro.txt"
    f = open(macro_file_path,'r')
    result = False
    for lines in f.readlines():
        split_line = lines.split(' ')

        if int(split_line[0]) != _file_num:
            continue

        if int(split_line[1]) <= _frame_num <= int(split_line[2]):
            result = True

    f.close()
    return result


"""
# TODO: using verb xml file
def check_verb_file(_file_num, _frame_num):
    verb_file_path = ""
"""


def make_preprocess_data(_xml_path, _json_path, _save_path, _using_macro=False):
    for file in files:
        xml_file_path = _xml_path + "\\%03d.xml" % file

        tree = parse(xml_file_path)
        root = tree.getroot().find('Objects')
        for object in root.findall('Object'):
            if not int(object.find('Type').text) in [1, 111]:
                continue

            tracks = object.find('Tracks')
            for track in tracks.findall('Track'):
                attr = track.attrib
                json_file_path = _json_path + "\\%03d\\%03d_%012d_keypoints.json" % (file, file, int(attr['frameNum']))

                people = read_pose_(json_file_path)['people']
                for person in people:
                    key_point = person['pose_keypoints']

                    if not (int(attr['X']) <= key_point[3] <= int(attr['X']) + int(attr['W']) and \
                                            int(attr['Y']) <= key_point[4] <= int(attr['Y']) + int(attr['H'])):
                        continue

                    label = 0
                    if int(object.find('Type').text) == 111:
                        if _using_macro:
                            if check_macro_file(file, int(attr['frameNum'])):
                                label = 1
                        """
                        else:
                            if check_verb_file(file, int(attr['frameNum'].text)):
                                label = 1
                        """

                    data = []
                    data.append(object.find('ID').text)
                    data.append(object.find('Type').text)
                    data.append(attr['frameNum'])

                    norm_key_point = normalize_pose_(key_point)
                    for i in range(18):
                        data.append(str(norm_key_point[i * 3]))
                        data.append(str(norm_key_point[i * 3 + 1]))
                    data.append(str(label))

                    file_name = "%06d.txt" % file
                    if file_name in os.listdir(_save_path):
                        txt_path = _save_path + "\\%s" % file_name
                        f = open(txt_path, 'a')

                        iter = 1
                        for dat in data:
                            f.write(dat)

                            if len(data) == iter:
                                f.write("\n")
                                continue

                            f.write(",")
                            iter += 1
                        f.close()

                    else:
                        txt_path = _save_path + "\\%s" % file_name
                        f = open(txt_path, 'w')

                        iter = 1
                        for dat in data:
                            f.write(dat)

                            if len(data) == iter:
                                f.write("\n")
                                continue

                            f.write(",")
                            iter += 1
                        f.close()


def make_macro_file_to_dict():
    macro_path = "C:\Users\JM\Desktop\Data\ETRIrelated\pose classification\class1macro.txt"
    f = open(macro_path, 'r')

    littering_info = {}
    for lines in f.readlines():
        split_line = lines.split(" ")
        littering_info[int(split_line[0])] = [int(split_line[1]), int(split_line[2])]

    f.close()
    return littering_info


def data_loader(_data_dir_path, _interval_size, _step_size, _posi_threshold):
    f = open(_data_dir_path, 'r')

    action_data = []
    data = {}
    for lines in f.readlines():
        split_line = lines.split(',')

        if not int(split_line[0]) in data.keys():
            data[int(split_line[0])] = {}

        data[int(split_line[0])][int(split_line[2])] = []
        split_data = split_line[3:39]
        for i in range(len(split_data)):
            data[int(split_line[0])][int(split_line[2])].append(float(split_data[i]))
        data[int(split_line[0])][int(split_line[2])].append(int(split_line[-1]))
    f.close()

    for person_id in data.keys():
        frame_key = data[person_id].keys()
        frame_key.sort()
        if len(frame_key) < _interval_size:
            continue
        start = 0
        end = _interval_size
        while 1:
            if end >= len(frame_key):
                break

            if frame_key[end] != frame_key[start] + _interval_size:
                break

            label_check = 0
            action_data.append([])
            for i in frame_key[start:end]:
                for j in range(36):
                    action_data[-1].append(data[person_id][i][j])

                if data[person_id][i][-1] == 1:
                    label_check += 1

            if label_check > _posi_threshold:
                action_data[-1].append(1)

            else:
                action_data[-1].append(0)

            start += _step_size
            end += _step_size

    return action_data


def normalize_pose_(_pose_data):

    neck_x = _pose_data[3]
    neck_y = _pose_data[4]
    base_index = 0

    while base_index < 18:
        _pose_data[base_index*3] -= neck_x
        _pose_data[base_index*3+1] -= neck_y
        base_index += 1

    return _pose_data


def support_vector_machine_classifier_(train_data, train_class, test_data):
    from sklearn.svm import SVC

    return SVC(kernel='linear', C=0.1).fit(train_data, train_class).predict(test_data)

if __name__ == '__main__':
    """
    xml_dir_path = "C:\\Users\JM\\Desktop\Data\\ETRIrelated\\final_xml"
    json_dir_path = "D:\\etri_data\\jsonfile_class1"
    save_dir_path = "C:\Users\JM\Desktop\Data\ETRIrelated\preprocess_data"

    make_preprocess_data(xml_dir_path, json_dir_path, save_dir_path, True)
    """

    # parameters
    data = []
    threshold = 10
    interval = 20
    step = 5

    for file in files:

        data_dir = "C:\\Users\\JM\\Desktop\\Data\\ETRIrelated\\preprocess_data\\%06d.txt" % file
        if files.index(file) == 0:
            data = data_loader(data_dir, interval, step, threshold)
        else:
            tmp = data_loader(data_dir, interval, step, threshold)
            data.extend(tmp)

    skf = StratifiedKFold(n_splits=10)
    X = []
    y = []
    for dat in data:
        X.append(dat[0: 36*interval])
        y.append(dat[36*interval])

    X = np.asarray(X)
    y = np.asarray(y)
    precision = 0
    recall = 0
    for train_index, test_index in skf.split(X, y):

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
    
        predict_label = support_vector_machine_classifier_(X_train, y_train, X_test)

        result = precision_recall_fscore_support(y_test, predict_label, average='binary')
        precision += result[0]
        recall += result[1]

    print("precision: %f" % (precision/10))
    print("recall: %f" % (recall / 10))


