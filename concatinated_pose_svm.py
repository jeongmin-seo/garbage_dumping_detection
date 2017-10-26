# -*- coding: utf-8 -*-

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


class DataLoader:

    def __init__(self, _json_dir_path, _xml_dir_path, _save_dir_path, _file_num_list):
        self._json_dir_path = _json_dir_path
        self._xml_dir_path = _xml_dir_path
        self._save_dir_path = _save_dir_path
        self._file_num_list = _file_num_list

    def read_json_pose_(self, _file_num, _frame_num):
        json_file_path = self._json_dir_path + "\\%03d\\%03d_%012d_keypoints.json" % (_file_num, _file_num, _frame_num)
        f = open(json_file_path, 'r')
        js = json.loads(f.read())
        f.close()

        return js

    @staticmethod
    def check_pose_in_gtbox_(_key_point, _attr, _margin=0):
        if int(_attr['X']) - _margin <= _key_point[3] <= int(_attr['X']) + int(_attr['W']) + _margin and \
                int(_attr['Y']) - _margin <= _key_point[4] <= int(_attr['Y']) + int(_attr['H']) + _margin:
            return True

        else:
            return False


    @staticmethod
    def packaging_preprocess_data_(_key_point, _label, _object, _attr, _normalize, _scaling):
        data = []
        point = _key_point
        data.append(_object.find('ID').text)
        data.append(_object.find('Type').text)
        data.append(_attr['frameNum'])

        if _normalize:
            point = normalize_pose_(point)

        if _scaling:
            point = scaling_data_(point)

        for i in range(18):
            data.append(str(point[i * 3]))
            data.append(str(point[i * 3 + 1]))
        data.append(str(_label))

        return data

    def saving_preprocess_data_(self, _list_data, _file_num):
        file_name = "%06d.txt" % _file_num
        save_file_path = self._save_dir_path + "\\%s" % file_name

        if file_name in os.listdir(self._save_dir_path):
            f = open(save_file_path, 'a')

        else:
            f = open(save_file_path, 'w')

        iter = 1
        for dat in _list_data:
            f.write(dat)

            if len(_list_data) == iter:
                f.write("\n")
                continue

            f.write(",")
            iter += 1

        f.close()

    def preprocess_data_(self, _ground_truth="macro", _nomalize=True, _scaling=True):
        for file_number in self._file_num_list:
            xml_file_path = self._xml_dir_path + "\\%03d.xml" % file_number

            tree = parse(xml_file_path)
            objects = tree.getroot().find('Objects')
            for object in objects:
                if not int(object.find('Type').text) in [1, 111]:
                    continue

                tracks = object.find('Tracks')
                for track in tracks.findall('Track'):
                    attr = track.attrib

                    people = self.read_json_pose_(file_number, int(attr['frameNum']))['people']
                    for person in people:
                        key_point = person['pose_keypoints']

                        if not self.check_pose_in_gtbox_(key_point, attr):
                            continue

                        label = 0
                        if int(object.find('Type').text) == 111:
                            if _ground_truth == "macro":
                                if check_macro_file(file_number, int(attr['frameNum'])):
                                    label = 1
                            """
                            else:
                                if check_verb_file(file, int(attr['frameNum'].text)):
                                    label = 1
                            """

                        packaging_data = \
                            self.packaging_preprocess_data_(key_point, label, object, attr, _nomalize, _scaling)
                        self.saving_preprocess_data_(packaging_data, file_number)

    def load_data_(self, _interval_size, _step_size, _posi_threshold):

        action_data = []
        for file_name in os.listdir(self._save_dir_path):

            if int(file_name.split(".")[0]) not in self._file_num_list:
                continue

            _data_dir_path = self._save_dir_path + "\\" + file_name
            f = open(_data_dir_path, 'r')

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

    """
    @staticmethod
    def packaging_load_data_(_read_data, _interval_size, _step_size, _posi_threshold):

        action_data = []
        for person_id in _read_data.keys():
            frame_key = _read_data[person_id].keys()
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
                        action_data[-1].append(_read_data[person_id][i][j])

                    if _read_data[person_id][i][-1] == 1:
                        label_check += 1

                if label_check > _posi_threshold:
                    action_data[-1].append(1)

                else:
                    action_data[-1].append(0)

                start += _step_size
                end += _step_size

        return action_data
    """


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


########################################################################
#              normalize the data using neck coordinate                #
########################################################################
def normalize_pose_(_pose_data):

    neck_x = _pose_data[3]
    neck_y = _pose_data[4]
    base_index = 0

    while base_index < 18:
        _pose_data[base_index*3] -= neck_x
        _pose_data[base_index*3+1] -= neck_y  # 목좌표로 좌표계 변환
        base_index += 1

    return _pose_data


########################################################################
#            scaling the data using knee & ankle distance              #
########################################################################
def scaling_data_(_pose_data):

    light_knee, light_ankle = [_pose_data[36], _pose_data[37]], [_pose_data[39], _pose_data[40]]
    right_knee, right_ankle = [_pose_data[27], _pose_data[28]], [_pose_data[30], _pose_data[31]]
    base_index = 0

    right_dist = ((right_knee[0] - right_ankle[0]) ** 2 + (right_knee[1] - right_ankle[1]) ** 2) ** 0.5  # 오른쪽 무릎부터 발목까지 거리
    light_dist = ((light_knee[0] - light_ankle[0]) ** 2 + (light_knee[1] - light_ankle[1]) ** 2) ** 0.5  # 왼쪽 무릎부터 발목까지 거리
    dist = right_dist if right_dist > light_dist else light_dist                                         # 오른쪽 왼쪽중 더 긴 거리 선택

    while base_index < 18:
        _pose_data[base_index*3] /= dist                                                                 # 선택된 거리를 1로 하는 좌표들로 변환
        _pose_data[base_index*3+1] /= dist
        base_index += 1

    return _pose_data


def support_vector_machine_classifier_(train_data, train_class, test_data):
    from sklearn.svm import SVC

    return SVC(kernel='linear', C=0.1).fit(train_data, train_class).predict(test_data)


if __name__ == '__main__':

    xml_dir_path = "C:\\Users\JM\\Desktop\Data\\ETRIrelated\\final_xml"
    json_dir_path = "D:\\etri_data\\jsonfile_class1"
    save_dir_path = "C:\Users\JM\Desktop\Data\ETRIrelated\preprocess_data"

    # parameters
    data = []
    threshold = 10
    interval = 20
    step = 5

    loader = DataLoader(json_dir_path, xml_dir_path, save_dir_path, files)
    loader.preprocess_data_(_scaling=False)
    data = loader.load_data_(interval, step, threshold)

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
    print set(y)
    for train_index, test_index in skf.split(X, y):

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
    
        predict_label = support_vector_machine_classifier_(X_train, y_train, X_test)

        result = precision_recall_fscore_support(y_test, predict_label, average='binary')
        precision += result[0]
        recall += result[1]

    print("precision: %f" % (precision/10))
    print("recall: %f" % (recall / 10))


