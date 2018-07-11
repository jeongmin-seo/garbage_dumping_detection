# -*- coding: utf-8 -*-

import json
import os
from xml.etree.ElementTree import ElementTree, parse, dump, Element, SubElement
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.svm import SVC
import copy
from sklearn.model_selection import GridSearchCV

grid_params_ = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4, 'auto'], 'C': [1, 10, 100, 1000]},
                {'kernel': ['linear'], 'C': [1, 10, 100, 1000]},
                {'kernel': ['poly'], 'gamma':[1e-3, 1e-4, 'auto'],
                 'C': [1, 10, 100, 1000], 'degree': [2, 3, 4, 5]},
                {'kernel': ['sigmoid'], 'C': [1, 10, 100, 1000], 'gamma':[1e-3, 1e-4, 'auto']}]

params = {'step':      1,
          'interval':  1,
          'threshold': 0,
          'posi_label': 1,
          'bDrawGraph': True,
          'bUsingDisparity': False,
          'bNorm': False
          }
"""
# bending pose
files = [5, 15, 17, 18, 28,
         31, 41, 42, 58, 100,
         113, 115, 117, 124,
         125, 127, 133, 136, 140,
         144, 147,  160, 161,
         164, 165, 172, 186, 191,
         196, 199, 202, 207, 213
         ]
# 153 120 172,
frame = [703, 319, 278, 224, 755,
         1037, 871, 1442, 1761, 288,
         170, 269, 1049, 214,
         408, 499, 364, 202, 329,
         359, 254,  314, 135,
         369, 269, 839, 628, 522,
         715, 1194, 176, 352, 252]
# 419, 99 839,
"""

"""
# class 2
files = [6, 23, 13, 46, 57,
         61, 69, 95, 99]

frame = [593, 509, 329, 639, 824,
         134, 486, 319, 599]
"""


# class 3
files = [214, 195, 192, 188, 187,
         184, 183, 157, 123]

frame = [344, 271, 279, 367, 556,
         1596, 905, 409, 219]

# info 형태 [file_num, id, 시작 frame, 끝 frame]


class Visualizer:

    def __init__(self, _sample_info, _last_frame_num):
        self.sample_info = _sample_info
        self.last_frame_num = _last_frame_num
        self.graph_save = True
        self.video_save = False
        self.all_dict = {}
        for i in set(self.sample_info[:, 0]):
            self.all_dict[i] = {}

    def check_true_posi_frame(self, _predict_label, _test_label, _test_index, _posi):
        for index, test_idx in enumerate(_test_index):
            if _predict_label[index] == _test_label[index] and _predict_label[index] == _posi:
                cur_info = self.sample_info[test_idx]
                self.true_posi_frame[cur_info[0]].append(cur_info[2])

    def making_graph_(self, _all_dict, _ground_truth):
        print("Start Drawing Graph")

        red_patch = mpatches.Patch(color='red', label='FN')
        yellow_patch = mpatches.Patch(color='yellow', label='FP')
        gray_patch = mpatches.Patch(color='darkgray', label='TN')
        green_patch = mpatches.Patch(color='green', label='TP')
        for file_number in _all_dict.keys():
            for person in _all_dict[file_number].keys():
                frame_length = frame[files.index(file_number)]
                height = len(_all_dict[file_number][person])
                fig = plt.figure()
                ax = fig.add_subplot(1, 1, 1)
                ax.set_xlim([0, frame_length + 1])
                ax.set_ylim([0, len(_all_dict[file_number][person]) + 1])

                if file_number in _ground_truth.keys() and person in _ground_truth[file_number].keys():
                    gt_boundary = _ground_truth[file_number][person]
                    for gt in gt_boundary:
                        plt.bar(gt[0], height + 1, color="lightblue", width=gt[1] - gt[0], align='edge')

                for i, result_info in enumerate(_all_dict[file_number][person]):
                    xs = range(result_info[0], result_info[1] + 1)
                    ys = [i + 1] * len(xs)

                    if result_info[2] == "true_positive":
                        plt.plot(xs, ys, color="green")

                    elif result_info[2] == "true_negative":
                        plt.plot(xs, ys, color="darkgray")

                    elif result_info[2] == "false_positive":
                        plt.plot(xs, ys, color="yellow")

                    else:
                        plt.plot(xs, ys, color="red")

                plt.legend(handles=[red_patch, yellow_patch, gray_patch, green_patch])
                plt.tight_layout()

                plt.savefig('file_%d_id_%d' % (file_number, person))
                plt.close(fig)
        pass

    def save_video_(self):
        pass


class DataLoader:

    def __init__(self, _json_dir_path, _xml_dir_path, _save_dir_path, _file_num_list):  # ,interpolation_size):
        self.json_dir_path = _json_dir_path
        self.xml_dir_path = _xml_dir_path
        self.save_dir_path = _save_dir_path
        self.file_num_list = _file_num_list
        # self.interpolation_size = _interpolation_size  # interpolation 할 때 앞뒤 몇 frame 을 볼지

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

        if (int(_attr['X']) - _margin <= _key_point[3] <= int(_attr['X']) + int(_attr['W']) + _margin and
                int(_attr['Y']) - _margin <= _key_point[4] <= int(_attr['Y']) + int(_attr['H']) + _margin) and \
                ((int(_attr['X']) - _margin <= _key_point[6] <= int(_attr['X']) + int(_attr['W']) + _margin and
                 int(_attr['Y']) - _margin <= _key_point[7] <= int(_attr['Y']) + int(_attr['W']) + _margin) or \
                int(_attr['X']) - _margin <= _key_point[15] <= int(_attr['X']) + int(_attr['W']) + _margin and
                 int(_attr['Y']) - _margin <= _key_point[16] <= int(_attr['Y']) + int(_attr['W']) + _margin):
            return True

        else:
            return False

    # @staticmethod
    def packaging_preprocess_data_(self, _key_point, _label, _object, _attr, _normalize, _scaling): #, _normalize, _scaling):
        result_data = []
        point = _key_point
        result_data.append(_object.find('ID').text)
        result_data.append(_object.find('Type').text)
        result_data.append(_attr['frameNum'])


        # interpolation 부분 넣기
        if _normalize:
            point = self.normalize_pose_(point)

        if _scaling:
            point = self.scaling_data_(point)

        for i in range(18):
            result_data.append(str(point[i * 3]))
            result_data.append(str(point[i * 3 + 1]))
            result_data.append(str(point[i * 3 + 2]))
        result_data.append(str(_label))

        return result_data
    
    # labeling 된 데이터 저장하는 부분
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

    def preprocess_data_(self, _nomalize, _scaling, _ground_truth="macro"): # _nomalize, _scaling, _ground_truth="macro"):
        for file_number in self.file_num_list:
            xml_file_path = self.xml_dir_path + "\\%03d.xml" % file_number

            # xml 파일 읽어오기
            tree = parse(xml_file_path)
            objects = tree.getroot().find('Objects')
            for object in objects:
                if not int(object.find('Type').text) in [1, 111]:
                    continue

                tracks = object.find('Tracks')
                for track in tracks.findall('Track'):
                    attr = track.attrib

                    # key point 읽어오기
                    people = self.read_json_pose_(file_number, int(attr['frameNum']))['people']

                    # 사람별 key point 접근
                    for person in people:
                        key_point = person['pose_keypoints']

                        # 주요한 점들이 박스 밖으로 나간 경우에는 해당 Pose 를 없애기
                        if not self.check_pose_in_gtbox_(key_point, attr):
                            continue

                        label = 0
                        if int(object.find('Type').text) == 111:
                            if _ground_truth == "macro":
                                if check_macro_file(file_number, int(attr['frameNum']), int(object.find('ID').text)):  # positive frame 확인
                                    label = 3
                            """
                            else:
                                if check_verb_file(file, int(attr['frameNum'].text)):
                                    label = 1
                            """

                        packaging_data = \
                            self.packaging_preprocess_data_(key_point, label, object, attr ,_nomalize, _scaling) #, _nomalize, _scaling)
                        self.saving_preprocess_data_(packaging_data, file_number)


########################################################################
#                     interval 단위로  데이터 로드                       #
########################################################################
    def load_data_(self, _file_path):  # 데이터를 로드할 때 interval 단위의 데이터로 생성

        action_data = []
        data_info = []
        for file_name_ in os.listdir(_file_path):

            split_name = file_name_.split('.')

            if split_name[1] != 'txt':
                continue

            _data_dir_path = _file_path + "\\" + file_name_
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

            tmp_data, tmp_info = self.packaging_load_data_(data, int(file_name_.split(".")[0]))
            if not action_data:
                action_data = tmp_data
                data_info = tmp_info
                continue
            action_data.extend(tmp_data)
            data_info.extend(tmp_info)

        return action_data, data_info

    def packaging_load_data_(self, _read_data, _file_number):  # dictionary로 생성된 데이터를 interval 단위로 묶음

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
                    start += 1
                    end += 1
                    continue
                    # break

                # sample 정보 저장(file number, pose 시작 frame number, pose 끝 frame number
                sample_info.append([_file_number, person_id, frame_key[start], frame_key[end]])

                label_check = 0
                action_data.append([])
                for i in frame_key[start:end]:
                    """
                    print(len(_read_data[person_id][i]))
                    tmp_list = copy.deepcopy(_read_data[person_id][i])
                    tmp_list = self.normalize_pose_(tmp_list)

                    if params['bNorm']:
                        tmp_list = self.scaling_data_(tmp_list)
                    """
                    for j in range(36):
                        action_data[-1].append(tmp_list[j])

                    if params['bUsingDisparity']:
                        if i == frame_key[start]:
                            for m in range(36):
                                action_data[-1].append(0)

                        else:
                            for k in range(36):
                                action_data[-1].append(_read_data[person_id][i][k] - _read_data[person_id][i - 1][k])

                    if _read_data[person_id][i][-1] == 1:
                        label_check += 1

                """
                if params['bUsingDisparity']:
                    for i in frame_key[start:end]:

                        if i == frame_key[start]:
                            for j in range(36):
                                action_data[-1].append(0)

                        else:
                            for j in range(36):
                                action_data[-1].append(_read_data[person_id][i][j] - _read_data[person_id][i-1][j])
                """
                if label_check > params['threshold']:
                    action_data[-1].append(1)

                else:
                    action_data[-1].append(0)

                start += params['step']
                end += params['step']

        return action_data, sample_info

    ########################################################################
    #              normalize the data using neck coordinate                #
    ########################################################################
    @staticmethod
    def normalize_pose_(_pose_data):

        neck_x = _pose_data[2]
        neck_y = _pose_data[3]
        # base_index = 0

        for base_index in range(18):
            _pose_data[base_index * 2] -= neck_x
            _pose_data[base_index * 2 + 1] -= neck_y  # 목좌표로 좌표계 변환

        """
        # print(len(_pose_data))
        while base_index < 18:
            if _pose_data[base_index * 2] == 0 and _pose_data[base_index * 2 + 1] == 0:
                base_index += 1
                continue

            _pose_data[base_index * 2] -= neck_x
            _pose_data[base_index * 2 + 1] -= neck_y  # 목좌표로 좌표계 변환
            base_index += 1
        """
        return _pose_data

    ########################################################################
    #           scaling the data using neck to shoulder distance           #
    ########################################################################
    @staticmethod
    def scaling_data_(_pose_data):

        neck = [_pose_data[2], _pose_data[3]]
        # right_shoulder = [_pose_data[4], _pose_data[5]]
        # left_shoulder = [_pose_data[10], _pose_data[11]]
        right_factor = [_pose_data[16], _pose_data[17]]
        left_factor = [_pose_data[22], _pose_data[23]]

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


def check_macro_file(_file_num, _frame_num, _person_id):
    macro_file_path = "C:\\Users\\JM\\Desktop\\Data\\ETRIrelated\\pose classification\\class3macro.txt"
    f = open(macro_file_path, 'r')
    result = False
    for lines in f.readlines():
        split_line = lines.split(' ')

        if int(split_line[0]) != _file_num:
            continue

        if int(split_line[-1]) != _person_id:
            continue

        if int(split_line[1]) <= _frame_num <= int(split_line[2]):
            result = True

    f.close()
    return result


def support_vector_machine_classifier_(train_data, train_class):
    from sklearn.svm import SVC

    return SVC(kernel='rbf', C=1).fit(train_data, train_class)


def drawing_graph_(_all_dict, _ground_truth):

    print("Start Drawing Graph")

    red_patch = mpatches.Patch(color='red', label='FN')
    yellow_patch = mpatches.Patch(color='yellow', label='FP')
    gray_patch = mpatches.Patch(color='darkgray', label='TN')
    green_patch = mpatches.Patch(color='green', label='TP')
    for file_number in _all_dict.keys():
        if file_number != 172:
            continue

        for person in _all_dict[file_number].keys():
            frame_length = frame[files.index(file_number)]
            height = len(_all_dict[file_number][person])
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.set_xlim([0, frame_length+1])
            ax.set_ylim([0, len(_all_dict[file_number][person])+1])

            if file_number in _ground_truth.keys() and person in _ground_truth[file_number].keys():
                gt_boundary = _ground_truth[file_number][person]
                for gt in gt_boundary:
                    plt.bar(gt[0], height+1, color="lightblue", width=gt[1]-gt[0], align='edge')

            for i, result_info in enumerate(_all_dict[file_number][person]):
                xs = range(result_info[0], result_info[1]+1)
                ys = [i+1] * len(xs)

                if result_info[2] == "true_positive":
                    plt.plot(xs, ys, color="green")

                elif result_info[2] == "true_negative":
                    plt.plot(xs, ys, color="darkgray")

                elif result_info[2] == "false_positive":
                    plt.plot(xs, ys, color="yellow")

                else:
                    plt.plot(xs, ys, color="red")

            plt.legend(handles=[red_patch, yellow_patch, gray_patch, green_patch])
            plt.tight_layout()

            plt.savefig('file_%d_id_%d' % (file_number, person))
            plt.close(fig)


# 없애는 쪽으로 수정해야하는 함수
def read_gt_():
    macro_file_path = "C:\\Users\\JM\\Desktop\\Data\\ETRIrelated\\pose classification\\class1macro.txt"
    f = open(macro_file_path, 'r')
    gt_dict = {}
    for lines in f.readlines():
        split_line = lines.split(' ')
        file_number = int(split_line[0])
        person_id = int(split_line[-1])
        if file_number not in gt_dict.keys():
            gt_dict[file_number] = {}

        if person_id not in gt_dict[file_number].keys():
            gt_dict[file_number][person_id] = []
        gt_dict[file_number][person_id].append([int(split_line[1]), int(split_line[2])])

    f.close()
    return gt_dict


def save_model_to_xml_(_model):
    storage = Element("opencv_storage")
    my_svm = Element("my_svm")
    my_svm.attrib["type_id"] = "opencv-ml-svm"
    storage.append(my_svm)

    svm_type = SubElement(my_svm, "svm_type")
    svm_type.text = "C_SVC"

    kernel = SubElement(my_svm, "kernel")
    type   = SubElement(kernel, "type")
    type.text = str(_model.kernel)
    gamma = SubElement(kernel, "gamma")
    gamma.text = str(_model.gamma)

    C = SubElement(my_svm, "C")
    C.text = str(_model.C)

    term_criteria = SubElement(my_svm, "term_criteria")
    epsilon = SubElement(term_criteria, "epsilon")
    epsilon.text = str(_model.tol)                       # Dummy data because it is not necessary
    iterations = SubElement(term_criteria, "iterations")
    iterations.text = "1000"                        # Dummy data because it is not necessary

    var_all = SubElement(my_svm, "var_all")
    var_all.text = str(len(_model.support_vectors_[0]))
    var_count = SubElement(my_svm, "var_count")
    var_count.text = str(len(_model.support_vectors_[0]))

    class_count = SubElement(my_svm, "class_count")
    class_count.text = str(len(_model.n_support_))

    class_labels = SubElement(my_svm, "class_labels")
    class_labels.attrib["type_id"] = "opencv-matrix"
    rows = SubElement(class_labels, "rows")
    cols = SubElement(class_labels, "cols")
    dt = SubElement(class_labels, "dt")
    data = SubElement(class_labels, "data")
    rows.text = "2"
    cols.text = "1"
    dt.text = "i"
    data.text = "0 1"                         # TODO: Input SVM related

    sv_total = SubElement(my_svm, "sv_total")
    sv_total.text = str(len(_model.support_vectors_))

    support_vectors = SubElement(my_svm, "support_vectors")
    # str_vector = ""
    for i, vector in enumerate(_model.support_vectors_):
        tmp = " ".join([str(vec) for vec in vector])
        support_under_bar_ = SubElement(support_vectors, "_")
        support_under_bar_.text = tmp

    # support_vectors.text = str_vector

    decision_functions = SubElement(my_svm, "decision_functions")
    under_bar = SubElement(decision_functions, "_")
    sv_count = SubElement(under_bar, "sv_count")
    rho = SubElement(under_bar, "rho")
    alpha = SubElement(under_bar, "alpha")
    # index = SubElement(under_bar, "index")

    sv_count.text = str(len(_model.support_vectors_))
    rho.text = str(float(_model.intercept_))
    for i, vector in enumerate(_model.dual_coef_):
        tmp = " ".join([str(vec) for vec in vector])
        # support_under_bar_ = SubElement(support_vectors, "_")
        alpha.text = tmp
    # index.text = "0 1 2 3 4"                   # TODO: Input SVM related

    indent(storage)
    dump(storage)
    ElementTree(storage).write("model.xml")


def indent(elem, level=0):
    i = "\n" + level*"  "

    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        if level == 0:
            elem.text = '\n'

        for elem in elem:
            indent(elem, level+1)
        if not elem.tail or not elem.tail.strip():
            if level == 1:
                elem.tail = '\n'
            else:
                elem.tail = i

    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i


def make_opencv_data(_data, _label):
    f = open('class1.txt', 'w')

    l = 0
    for i, dat in enumerate(_data):
        for element in dat:
            tmp = '%f,' % element
            f.write(tmp)

        tmp = '%d\n' % y[i]
        f.write(tmp)

    f.close()
    print(l)

if __name__ == '__main__':

    # read data
    xml_dir_path = "C:\\Users\\JM\\Desktop\\Data\\ETRIrelated\\final_xml"
    json_dir_path = "D:\\etri_data\\pose"
    save_dir_path = "C:\\Users\\JM\\Desktop\\Data\\ETRIrelated\\preprocess_data"

    loader = DataLoader(json_dir_path, xml_dir_path, save_dir_path, files)

    if not os.listdir(save_dir_path):
        loader.preprocess_data_(_nomalize=False, _scaling=False)  # _nomalize=True, _scaling=True)

    """
    data, all_info = loader.load_data_(save_dir_path)

    # skf = StratifiedKFold(n_splits=10)
    X = []
    y = []
    for dat in data:
        if params['bUsingDisparity']:
            X.append(dat[0: 36*params['interval']*2])
            y.append(dat[36*params['interval']*2])

        else:
            X.append(dat[0: 36 * params['interval']])
            y.append(dat[36 * params['interval']])
    """
    """
    print len(X)
    print len(X[0])
    print X[0]
    print all_info[0]
    print len(y)
    print set(y)
    """
    # X = np.asarray(X)
    # y = np.asarray(y)

    # make_opencv_data(X, y)
    """
    all_dict = {}
    for info in all_info:
        file_num = info[0]
        person_id = info[1]

        if file_num not in all_dict.keys():
            all_dict[file_num] = {}

        if person_id not in all_dict[file_num].keys():
            all_dict[file_num][person_id] = []

        all_dict[file_num][person_id].append(info[2:4])

    all_info = np.asarray(all_info)

    precision = 0
    recall = 0
    all_predict = []
    test_all = []

    svc = SVC()
    clf = GridSearchCV(svc, grid_params_)
    clf.fit(X, y)
    print "\n\n"

    print clf
    print "\n\n"

    print clf.cv_results_
    print "\n\n"

    print clf.best_estimator_
    print "\n\n"

    print clf.best_params_
    print "\n\n"

    print clf.best_score_

    
    # 파일별로 학습하고 test 하는 코드
    for f_num in files:
        test_idx = []
        train_idx = []

        if f_num != 172:
            continue

        for i, f_info in enumerate(all_info):
            if f_info[0] == f_num:
                test_idx.append(i)
                continue
            train_idx.append(i)

        test_idx = np.asarray(test_idx)
        train_idx = np.asarray(train_idx)

        if not len(test_idx):
            print("error:", f_num)
            continue

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        info_test = all_info[test_idx]


        svc = SVC()
        clf = GridSearchCV(svc, grid_params_)
        clf.fit(X_train, y_train)
        print "\n\n"

        print clf
        print "\n\n"

        print clf.cv_results_
        print "\n\n"

        print clf.best_estimator_
        print "\n\n"

        print clf.best_params_
        print "\n\n"

        print clf.best_score_
        
        model = support_vector_machine_classifier_(X_train, y_train)

        
        print("result:",model.predict(X_test))
        print("C value:", model.__getattribute__('C'))
        print("gamma value:", model.__getattribute__("gamma"))
        print("params: ", model.get_params)
        save_model_to_xml_(model)
        
        # print(fit_svc.__getattribute__())
        # print(len(model.coef_[0]))
        # print(model)
        # print(model.dual_coef_)
        # print(len(model.dual_coef_))
        # print(len(model.dual_coef_[0]))
        
        predict_label = model.predict(X_test)
        print predict_label
        
        if not all_predict:
            all_predict = predict_label.tolist()
        else:
            all_predict.extend(predict_label.tolist())
        
        if not test_all:
            test_all = y_test.tolist()
        else:
            test_all.extend(y_test.tolist())

        for i, index in enumerate(test_idx):
            result_txt = ""
            if predict_label[i] == 1:
                if y_test[i] == predict_label[i]:
                    result_txt = "true_positive"
                else:
                    result_txt = "false_positive"

            else:
                if y_test[i] == predict_label[i]:
                    result_txt = "true_negative"

                else:
                    result_txt = "false_negative"

            file_num = info_test[i][0]
            person_id = info_test[i][1]
            idx = all_dict[file_num][person_id].index([info_test[i][2], info_test[i][3]])
            all_dict[file_num][person_id][idx].append(result_txt)

        result = precision_recall_fscore_support(y_test, predict_label, average='binary')
        print("file: ", f_num)
        print("precision: ", result[0], "recall: ", result[1])
        """

    # ground_truth = read_gt_()
    # drawing_graph_(all_dict, ground_truth)




