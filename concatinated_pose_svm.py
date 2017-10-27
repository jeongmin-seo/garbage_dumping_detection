# -*- coding: utf-8 -*-

import json
import os
from xml.etree.ElementTree import parse
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_recall_fscore_support
import numpy as np


params = {'step':      10,
          'interval':  30,
          'threshold': 25,
          'posi_label': 1
          }

files = [5, 15, 17, 18, 28,
         31, 41, 42, 58, 100,
         113, 115, 117, 120, 124,
         125, 127, 133, 136, 140,
         144, 147,  160, 161,
         164, 165, 172, 186, 191,
         196, 199, 202, 206, 213
         ]
# 153
frame = [703, 319, 278, 224, 755,
         1037, 871, 1442, 1761, 288,
         170, 269, 1049, 99, 214,
         408, 499, 364, 202, 329,
         359, 254,  314, 135,
         369, 269, 839, 628, 522,
         715, 1194, 176, 1028, 252]
# 419,

# info 형태 [frame_num, id, 시작 frame, 끝 frame]

class Visualizer:

    def __init__(self, _sample_info, _last_frame_num):
        self.sample_info = _sample_info
        self.last_frame_num = _last_frame_num
        self.graph_save = True
        self.video_save = False
        self.true_posi_frame = {}
        for i in set(self.sample_info[:, 0]):
            self.true_posi_frame[i] = []

    # 이부분 제대로 코딩 된건지 확인부터 해야함.
    def check_true_posi_frame(self, _predict_label, _test_label, _test_index, _posi):
        for index, test_idx in enumerate(_test_index):
            if _predict_label[index] == _test_label[index] and _predict_label[index] == _posi:
                cur_info = self.sample_info[test_idx]
                self.true_posi_frame[cur_info[0]].append(cur_info[2])

    def making_graph_(self):
        pass

    def save_video_(self):
        pass


class DataLoader:

    def __init__(self, _json_dir_path, _xml_dir_path, _save_dir_path, _file_num_list):
        self.json_dir_path = _json_dir_path
        self.xml_dir_path = _xml_dir_path
        self.save_dir_path = _save_dir_path
        self.file_num_list = _file_num_list

    def __del__(self):
        pass

    def read_json_pose_(self, _file_num, _frame_num):
        json_file_path = self.json_dir_path + "\\%03d\\%03d_%012d_keypoints.json" % (_file_num, _file_num, _frame_num)
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
        result_data = []
        point = _key_point
        result_data.append(_object.find('ID').text)
        result_data.append(_object.find('Type').text)
        result_data.append(_attr['frameNum'])

        if _normalize:
            point = normalize_pose_(point)

        if _scaling:
            point = scaling_data_(point)

        for i in range(18):
            result_data.append(str(point[i * 3]))
            result_data.append(str(point[i * 3 + 1]))
        result_data.append(str(_label))

        return result_data

    def saving_preprocess_data_(self, _list_data, _file_num):
        file_name = "%06d.txt" % _file_num
        save_file_path = self.save_dir_path + "\\%s" % file_name

        if file_name in os.listdir(self.save_dir_path):
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
        for file_number in self.file_num_list:
            xml_file_path = self.xml_dir_path + "\\%03d.xml" % file_number

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

    def load_data_(self):

        action_data = []
        data_info = []
        for file_name in os.listdir(self.save_dir_path):

            if int(file_name.split(".")[0]) not in self.file_num_list:
                continue

            _data_dir_path = self.save_dir_path + "\\" + file_name
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

            tmp_data, tmp_info = self.packaging_load_data_(data, int(file_name.split(".")[0]))
            if not action_data:
                action_data = tmp_data
                data_info = tmp_info
                continue
            action_data.extend(tmp_data)
            data_info.extend(tmp_info)

        return action_data, data_info

    @staticmethod
    def packaging_load_data_(_read_data, _file_number):

        action_data = []
        sample_info = []
        for person_id in _read_data.keys():
            frame_key = _read_data[person_id].keys()
            frame_key.sort()
            if len(frame_key) < params['interval']:
                continue
            start = 0
            end = params['interval']
            while 1:
                if end >= len(frame_key):
                    break

                if frame_key[end] != frame_key[start] + params['interval']:
                    break

                # sample 정보 저장(file number, pose 시작 frame number, pose 끝 frame number
                sample_info.append([_file_number, person_id, frame_key[start], frame_key[end]])

                label_check = 0
                action_data.append([])
                for i in frame_key[start:end]:
                    for j in range(36):
                        action_data[-1].append(_read_data[person_id][i][j])

                    if _read_data[person_id][i][-1] == 1:
                        label_check += 1

                if label_check > params['threshold']:
                    action_data[-1].append(1)

                else:
                    action_data[-1].append(0)

                start += params['step']
                end += params['step']

        return action_data, sample_info


def check_macro_file(_file_num, _frame_num):
    macro_file_path = "C:\\Users\\JM\\Desktop\\Data\\ETRIrelated\\pose classification\\class1macro.txt"
    f = open(macro_file_path, 'r')
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

    right_dist = ((right_knee[0] - right_ankle[0]) ** 2 + (right_knee[1] - right_ankle[1]) ** 2) ** 0.5
    light_dist = ((light_knee[0] - light_ankle[0]) ** 2 + (light_knee[1] - light_ankle[1]) ** 2) ** 0.5
    dist = right_dist if right_dist > light_dist else light_dist

    while base_index < 18:
        _pose_data[base_index*3] /= dist
        _pose_data[base_index*3+1] /= dist
        base_index += 1

    return _pose_data


def support_vector_machine_classifier_(train_data, train_class, test_data):
    from sklearn.svm import SVC

    return SVC(kernel='linear', C=0.1).fit(train_data, train_class).predict(test_data)


if __name__ == '__main__':

    # read data &
    xml_dir_path = "C:\\Users\JM\\Desktop\Data\\ETRIrelated\\final_xml"
    json_dir_path = "D:\\etri_data\\jsonfile_class1"
    save_dir_path = "C:\Users\JM\Desktop\Data\ETRIrelated\preprocess_data"

    loader = DataLoader(json_dir_path, xml_dir_path, save_dir_path, files)
    if not os.listdir(save_dir_path):
        loader.preprocess_data_(_scaling=False)

    data, info = loader.load_data_()

    skf = StratifiedKFold(n_splits=10)
    X = []
    y = []
    for dat in data:
        X.append(dat[0: 36*params['interval']])
        y.append(dat[36*params['interval']])

    X = np.asarray(X)
    y = np.asarray(y)
    info = np.asarray(info)  # data set 순서에 맞춰서 저장되어 있는 파일
    print(info[0])
    visualize = Visualizer(info, frame)
    """
    precision = 0
    recall = 0
    for train_index, test_index in skf.split(X, y):

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        info_train, info_test = info[train_index], info[test_index]
        
        predict_label = support_vector_machine_classifier_(X_train, y_train, X_test)

        result = precision_recall_fscore_support(y_test, predict_label, average='binary')
        precision += result[0]
        recall += result[1]

    print("precision: %f" % (precision/10))
    print("recall: %f" % (recall / 10))
    """