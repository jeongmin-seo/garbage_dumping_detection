import json
from sklearn.model_selection import StratifiedKFold
import cv2
import os
import glob
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from xml.etree.ElementTree import parse


#########################################################################
# files = [1, 12, 30, 31, 43, 67, 69]
# frame = [404, 732, 1619, 1037, 1906, 874, 486]
# Precision:0.8436331  Recall:0.4432278  Accuracy:0.9328258
#########################################################################

#########################################################################
# files = [5,14,19,30,33,38,44,189]
# frame = [703,453,2173,1619,1075,423,1083,287]
# Precision:  Recall:  Accuracy:
#########################################################################

#########################################################################
# files = [5,189]
# frame = [703,287]
#########################################################################

##########################################################################
files = [11, 12, 15, 17, 18, 28, 31,41, 42, 58, 113, 115, 117,
         120,124, 125, 127, 133, 136, 140, 144, 147, 153, 160, 161, 164, 165,
        172, 186, 189, 191, 196, 199, 202, 204, 213] #[1, 12, 30, 31, 43, 67, 69]
frame = [1030, 732, 319, 278, 224, 755, 1037, 871, 1442, 1761, 170, 269, 1049,
        99, 214, 408, 499, 364, 202, 329, 359, 254, 419, 314, 135, 369, 269,
         839,628,287,522,715, 1194, 176,744, 252] #[404, 732, 1619, 1037, 1906, 874, 486]
# Precision:0.8785148  Recall:0.3316766  Accuracy:0.8999
# Precision:0.870783  Recall:0.3367419  Accuracy:0.8981
# Precision:0.898605  Recall:0.3662108  Accuracy:0.909791
# 100 video 288 frame // 206video 1028 frames
###########################################################################

###########################################################################
# teahun's file
# files = [15,17,18,28,31,41,42]
# frame = [319, 278, 224, 755, 1037, 871, 1442]
###########################################################################

###########################################################################
# class 2
#files = [6, 13, 23, 46, 57, 61, 65, 69, 72, 95, 99]
#frame = [593, 329, 509, 639, 824, 134, 457, 486, 179, 319, 599]
###########################################################################

###########################################################################
# class 3
# files =[123,157,183,184,187,188,192,195,205,214]
# frame = [219, 409, 905, 1596, 556, 367, 279, 271, 340, 344]
###########################################################################

true_class_label_number = 1  # change true class label number
file_info = []
# kmeans_dict = {}
true_positive_dict = {}
# true_positive_frame = {}
true_sample_dict = {}
# true_sample_frame={}
frame_result_dict = {}
ground_truth_dict = {}
gt_and_detect_result = {}

for i in files:
    true_positive_dict[i] = {}
    true_sample_dict[i] = {}
    frame_result_dict[i] = []
    ground_truth_dict[i] = []


########################################################################
#                           read json file                             #
########################################################################
def read_pose_(filename):
    f = open(filename, 'r')
    js = json.loads(f.read())
    f.close()

    return js


########################################################################
#                read json and write python list data                  #
########################################################################
def import_data_(start_num, end_num, index):
    pose_key_points = []
    y = []
    negative_sample = []
    positive_sample = []
    file_number = start_num
    file_index = files[index]

    while file_number <= end_num:
        pose_file = "D:\etri_data\macrojson\macrojson\%03d\%03d_%012d_keypoints.json" \
                   % (file_index, file_index, file_number)
        # pose_file = "/home/jmseo/Desktop/ETRI/%03d/%03d_%012d_keypoints.json" % (file_index, file_index, file_number)
        # pose_file = "/home/jmseo/PycharmProjects/ETRIsvm/teahun/%03d/%03d_%012d_keypoints.json" \
        #             % (file_index, file_index, file_number)
        # class2
        #pose_file = "D:\ETRI\class2json\class2json\%03d\%03d_%012d_keypoints.json" \
        #            % (file_index, file_index, file_number)
        # class3
        # pose_file = "/home/jmseo/PycharmProjects/ETRIsvm/class3json/%03d/%03d_%012d_keypoints.json" \
        #             % (file_index, file_index, file_number)

        frame_result_dict[file_index].append(False)  # evaluation related

        dict_pose = read_pose_(pose_file)  # dict_pose = {}
        people = dict_pose['people']

        if not people:
            file_number += 1
            continue

        for pose_list in enumerate(people):
            # if pose_list[1]['pose_keypoints'][36] != 0 and pose_list[1]['pose_keypoints'][37] != 0 and \
                            # pose_list[1]['pose_keypoints'][39] != 0 and pose_list[1]['pose_keypoints'][40] != 0:

            if pose_list[1]['pose_keypoints'][29] != 0 and pose_list[1]['pose_keypoints'][32] != 0 and \
                    pose_list[1]['pose_keypoints'][38] != 0 and pose_list[1]['pose_keypoints'][41] != 0 and \
                    pose_list[1]['pose_keypoints'][5] != 0:

                if len(pose_list[1]['pose_keypoints']) == 54:
                    negative_sample.append(pose_list[1]['pose_keypoints'])
                    y.append(0)

                elif len(pose_list[1]['pose_keypoints']) == 55:
                    positive_sample.append(pose_list[1]['pose_keypoints'])
                    y.append(pose_list[1]['pose_keypoints'].pop())
                    if file_number not in true_sample_dict[file_index]:
                        true_sample_dict[file_index][file_number] = []
                    true_sample_dict[file_index][file_number].append(pose_list[0])

                else:
                    print(pose_file)

                pose_key_points.append(pose_list[1]['pose_keypoints'])
                file_info.append([file_index, file_number, pose_list[0]])

        file_number += 1

    return pose_key_points, y, positive_sample, negative_sample


########################################################################
#              normalize the data using neck coordinate                #
########################################################################
def normalize_pose_(pose_data):
    norm_pose_data = [pose for pose in pose_data]

    for pose in norm_pose_data:
        neck_x = pose[3]
        neck_y = pose[4]
        base_index = 0

        while base_index < 18:
            pose[base_index*3] -= neck_x
            pose[base_index*3+1] -= neck_y
            base_index += 1

    return norm_pose_data


########################################################################
#            scaling the data using knee & ankle distance              #
########################################################################
def scaling_data_(pose_data):
    scaling_pose_data = [pose for pose in pose_data]

    for pose in scaling_pose_data:
        light_knee, light_ankle = [pose[36], pose[37]], [pose[39], pose[40]]
        right_knee, right_ankle = [pose[27], pose[28]], [pose[30], pose[31]]
        base_index = 0

        right_dist = ((right_knee[0] - right_ankle[0]) ** 2 + (right_knee[1] - right_ankle[1]) ** 2) ** 0.5
        light_dist = ((light_knee[0] - light_ankle[0]) ** 2 + (light_knee[1] - light_ankle[1]) ** 2) ** 0.5
        dist = right_dist if right_dist > light_dist else light_dist

        while base_index < 18:
            pose[base_index*3] /= dist
            pose[base_index*3+1] /= dist
            base_index += 1

    return scaling_pose_data


########################################################################
#                 make the true positive result movie                  #
########################################################################
def check_true_positive_in_frame_(file_num, posi_dict, true_dict):
    file_path = "/home/jmseo/openpose/examples/media/ETRI1/videoresult/%03dresult.avi" % file_num

    cap = cv2.VideoCapture(file_path)
    fps, width, height = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    output = "%03d_check_positive.avi" % file_num

    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    out = cv2.VideoWriter(output, fourcc, fps, (int(width), int(height)))

    font = cv2.FONT_HERSHEY_SIMPLEX
    frame_num = 0

    while cap.isOpened():

        ret, frame = cap.read()
        if not ret:
            break

        cv2.putText(frame, str(frame_num), (100, 100), font, 4, (255, 255, 255), 2, cv2.LINE_AA)

        # add new code
        pose_file = "/home/jmseo/PycharmProjects/ETRIsvm/macrolabeling/macrojson/%03d/%03d_%012d_keypoints.json" \
                    % (file_num, file_num, frame_num)

        # pose_file = "/home/jmseo/PycharmProjects/ETRIsvm/result/%03d/%03d_%012d_keypoints.json" \
        #             % (file_num, file_num, frame_num)
        # pose_file = "/home/jmseo/PycharmProjects/ETRIsvm/teahun/%03d/%03d_%012d_keypoints.json" \
        #             % (file_num, file_num, frame_num)
        # pose_file = "/home/jmseo/PycharmProjects/ETRIsvm/throw/%03d/%03d_%012d_keypoints.json" \
        #             % (file_num, file_num, frame_num)
        dict_pose = read_pose_(pose_file)  # dict_pose={}
        people = dict_pose['people']

        if frame_num in true_dict:
            for i in true_dict[frame_num]:
                temp = people[i]['pose_keypoints']
                temp_x = [temp[x * 3] for x in range(0, 18)]
                temp_y = [temp[y * 3 + 1] for y in range(0, 18)]
                cv2.rectangle(frame,
                              (int(min(filter(lambda x: x > 0, temp_x))), int(min(filter(lambda x: x > 0, temp_y)))),
                              (int(max(temp_x)), int(max(temp_y))),
                              (255, 0, 0), thickness=3)

        if posi_dict and frame_num in posi_dict:
            cv2.circle(frame, (int(width)-200, int(height)-100), 40, (255, 255, 255), 40)

            # add new code
            pose_file = "/home/jmseo/PycharmProjects/ETRIsvm/macrolabeling/macrojson/%03d/%03d_%012d_keypoints.json" \
                        % (file_num, file_num, frame_num)

            # pose_file = "/home/jmseo/PycharmProjects/ETRIsvm/class2json/%03d/%03d_%012d_keypoints.json" \
            #             % (file_index, file_index, file_number)
            # pose_file = "/home/jmseo/PycharmProjects/ETRIsvm/teahun/%03d/%03d_%012d_keypoints.json" \
            #             % (file_num, file_num, frame_num)
            # pose_file = "/home/jmseo/PycharmProjects/ETRIsvm/throw/%03d/%03d_%012d_keypoints.json" \
            #             % (file_num, file_num, frame_num)
            dict_pose = read_pose_(pose_file)  # dict_pose={}
            people = dict_pose['people']

            for i in posi_dict[frame_num]:
                temp = people[i]['pose_keypoints']
                temp_x = [temp[x*3] for x in range(0, 18)]
                temp_y = [temp[y*3 + 1] for y in range(0, 18)]
                cv2.rectangle(frame,
                              (int(min(filter(lambda x: x > 0, temp_x))+5), int(min(filter(lambda x: x > 0, temp_y))+5)),
                              (int(max(temp_x)+5), int(max(temp_y)+5)),
                              (0, 0, 255), thickness=3)

            """
            for i in true_positive_frame[posi_list[0]]:
                try:
                    temp = people[i]['pose_keypoints']
                    temp_x = [temp[x*3] for x in range(0, 18)]
                    temp_y = [temp[y*3 + 1] for y in range(0, 18)]
                    cv2.rectangle(frame,
                                  (int(min(filter(lambda x: x > 0, temp_x))), int(min(filter(lambda x: x > 0, temp_y)))),
                                  (int(max(temp_x)), int(max(temp_y))),
                                  (0, 0, 255), thickness=3)

                except:
                    print("frame:", frame_num)
                    print(i, len(people))
                    print(true_positive_frame.keys())
                    print(true_positive_dict.keys())
            """

            # posi_list.pop(0)

        # frame_name = "%03d_check_positive_%06d.jpg" % (file_num, frame_num)
        # cv2.imwrite(frame_name, frame)
        out.write(frame)
        cv2.imshow('frame', frame)
        frame_num += 1

        if cv2.waitKey(int(fps)) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


########################################################################
#             calculate the Precision, Recall and Accuracy             #
########################################################################
def check_classified_result_(predict_class, test_class, test_index, true_class_num):
    false_positive = 0
    false_negative = 0
    true_positive = 0
    true_negative = 0
    for index in range(0, len(predict_class)):

        # change true class label
        if predict_class[index] == true_class_label_number:

            # Evaluation related code
            video_idx = file_info[test_index[index]][0]
            frame_idx = file_info[test_index[index]][1]

            # Evaluation related result save code
            frame_result_dict[video_idx][frame_idx] = True

            """
            if test_class[index] == true_class_label_number:
                true_positive += 1
#                true_positive_dict[file_info[test_index[index]][0]].append(file_info[test_index[index]][1])

#               if not file_info[test_index[index]][1] in true_positive_frame:
#                    true_positive_frame[file_info[test_index[index]][1]] = []

#                true_positive_frame[file_info[test_index[index]][1]].append(file_info[test_index[index]][2])

                if not file_info[test_index[index]][1] in true_positive_dict[file_info[test_index[index]][0]]:
                    true_positive_dict[file_info[test_index[index]][0]][file_info[test_index[index]][1]]=[]
                true_positive_dict[file_info[test_index[index]][0]][file_info[test_index[index]][1]].append(file_info[test_index[index]][2])

            elif test_class[index] == 0:
                false_positive += 1
            """
            if test_class[index] == 0:
                false_positive += 1

            else:
                true_positive += 1
                """
                if not file_info[test_index[index]][1] in true_positive_dict[file_info[test_index[index]][0]]:
                    true_positive_dict[file_info[test_index[index]][0]][file_info[test_index[index]][1]] = []
                true_positive_dict[file_info[test_index[index]][0]][file_info[test_index[index]][1]].append(
                    file_info[test_index[index]][2])
                """

        elif predict_class[index] == 0:
            if test_class[index] == 0:
                true_negative += 1

            else:  # elif test_class[index] == true_class_label_number:
                false_negative += 1
"""
    print('True Class: %d' % true_class_num,
          'True Positive: %d' % true_positive,
          'True Negative: %d' % true_negative,
          'False Positive: %d' % false_positive,
          'Missing: %d' % false_negative,
          'All: %d' % len(test_class))

    print('Precision: %f' % (float(true_positive)/(true_positive+false_positive)),
          'Recall: %f' % (float(true_positive)/(true_positive+false_negative)),
          'Accuracy: %f' % (float(true_positive + true_negative)/len(test_class)))
"""


########################################################################
#             predict class using support vector machine               #
########################################################################
def support_vector_machine_classifier_(train_data, train_class, test_data):
    from sklearn.svm import SVC

    return SVC(kernel='linear', C=0.1).fit(train_data, train_class).predict(test_data)


########################################################################
#            draw visualize graph /  compare GT with result            #
########################################################################
def visualize_classification_result_(frame_length, true_frame_list, file_number):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    xs = np.arange(frame_length)
    ys = [5] * frame_length

    # You can provide either a single color or an array. To demonstrate this,
    # the first bar of each set will be colored cyan.
    cs = ['darkblue'] * frame_length
    for cr in true_frame_list:
        cs[cr] = 'lightblue'
    ax.bar(xs, ys, zs=0, zdir='z', color=cs, width=1.1)

    for cr in true_frame_list:
        ys[cr] += 1

    y_plot = [0] * frame_length
    ax.plot(xs, ys=y_plot, zs=ys, color='darkred')

    ax.set_xlabel('time')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.title('%d file' % file_number)
    plt.savefig('%d file' % file_number)


########################################################################
#                         Sampling the data                            #
########################################################################
def random_sampling_negative_(negative_sample):
    import random

    return random.shuffle(negative_sample)


def read_ground_truth_(file_number):
    # read ground truth using xml parser

    ground_truth_list = [False for i in range(0, frame[files.index(file_number)])]

    file_name = "C:\Users\JM\Desktop\ETRI\ETRIxml\%03d.xml" % file_number
    # print(file_number)
    tree = parse(file_name)
    verbs = tree.getroot().find("Verbs").findall("Verb")

    for verb in verbs:
        Tracks = verb.find("Tracks").findall("Track")
        # print("Track legth", len(Tracks))
        for track in Tracks:
            ground_truth_list[int(track.get("frameNum"))] = True

    ground_truth_dict[file_number] = ground_truth_list

    return ground_truth_list


def overlap_window_(window_size_, predict_result_):

    result_dict = {}
    result_range_dict = {}
    for key in predict_result_.keys():
        result_dict[key] = []
        result_range_dict[key] = []
        size = len(predict_result_[key])
        for index in range(0, size):

            true_label_num = 0
            false_label_num = 0

            if window_size_ <= index <= size-window_size_-1:
                start = index - window_size_
                end = index + window_size_+1
                for i in range(start, end):

                    if predict_result_[key][i] == true_class_label_number:
                        true_label_num += 1

                    else:
                        false_label_num += 1

            if true_label_num == false_label_num:
                result_dict[key].append(predict_result_[key][index])

            elif true_label_num > false_label_num:
                result_dict[key].append(True)

            else:
                result_dict[key].append(False)

    return result_dict


def check_positive_range(_dictionary):
    result = {}
    cur_state = False
    start = 0
    end = 0
    for key in _dictionary.keys():

        result[key] = []
        for size in range(0, frame[files.index(key)]):

            if cur_state != _dictionary[key][size]:
                if cur_state :
                    cur_state = False
                    end = size
                    result[key].append([start, end])

                else:
                    cur_state = True
                    start = size

    return result


def detect_base_calculate_result_(_detect_range_list, _gt_and_detect_list):
    false_posi = 0
    for detect_range in _detect_range_list:

        if not detect_range:
            continue

        rate = float(_gt_and_detect_list[detect_range[0]: detect_range[1]].count(True)) \
               / float(detect_range[1] - detect_range[0])
        if rate < 0.5:
            false_posi += 1

    return false_posi


def gt_base_calculate_result_(_gt_range_list, _gt_and_detect_list):
    true_posi = 0
    false_nega = 0
    for gt_range in _gt_range_list:

        if not gt_range:
            continue

        rate = float(_gt_and_detect_list[gt_range[0]: gt_range[1]].count(True)) \
               / float(gt_range[1] - gt_range[0])

        if rate >= 0.5:
            true_posi += 1
        else:
            false_nega += 1

    return true_posi, false_nega


def calculate_evaluation_(_detect_result_dict, _ground_truth_dict):
    false_posi = 0
    true_posi = 0
    false_nega = 0
    _detect_result_range = check_positive_range(_detect_result_dict)
    _ground_truth_range = check_positive_range(_ground_truth_dict)

    for key in _detect_result_dict.keys():
        gt_and_detect_result[key] = [a and b for a, b in zip(_detect_result_dict[key], _ground_truth_dict[key])]

        tp, fn = gt_base_calculate_result_(_ground_truth_range[key], gt_and_detect_result[key])
        false_posi += detect_base_calculate_result_(_detect_result_range[key], gt_and_detect_result[key])
        true_posi += tp
        false_nega += fn

    print("True Positive:", true_posi)
    print("False Negative", false_nega)
    print("False Positive", false_posi)


def main():
    print('start')
    key_point = []
    pose_class = []


########################################################################
#                     import data to python list                       #
########################################################################
    for i in range(0, len(files)):
        tmp_key_point, tmp_pose_class, tmp_positive, tmp_negative = import_data_(0, frame[i], i)
        ground_truth_dict[files[i]] = read_ground_truth_(files[i])
        if i == 0:
            key_point = tmp_key_point
            pose_class = tmp_pose_class

        else:
            key_point.extend(tmp_key_point)
            pose_class.extend(tmp_pose_class)

########################################################################
#                      normalize & scaling data                        #
########################################################################

    norm_key_point = normalize_pose_(key_point)
    scaling_key_point = scaling_data_(norm_key_point)

########################################################################
#               extract body coordinate except confidence              #
########################################################################

    coord_key_point = []
    for point in scaling_key_point:
        coord_key_point.append([])
        for index in range(0, 18):
            coord_key_point[len(coord_key_point) - 1].append(point[index * 3])
            coord_key_point[len(coord_key_point) - 1].append(point[index * 3 + 1])

    true = 0
    for pose in pose_class:
        if pose == true_class_label_number:
            true += 1
    print("k-fold")


########################################################################
#       10-fold validation and split training & test data set          #
########################################################################

    # 10fold & shuffle = True
    skf = StratifiedKFold(n_splits=10, shuffle=True)

    print("all", len(coord_key_point))
    # split train test
    for train_index, test_index in skf.split(coord_key_point, pose_class):
        train_data = []
        test_data = []
        train_class = []
        test_class = []
        true_class_num = 0
        # print(len(train_index), len(test_index))
        for index in train_index:
            train_data.append(coord_key_point[index])
            train_class.append(pose_class[index])
        for index in test_index:
            test_data.append(coord_key_point[index])
            test_class.append(pose_class[index])
            if pose_class[index] == true_class_label_number:
                true_class_num += 1

########################################################################
#     predict test class & calculate Precision Recall and Accuracy     #
########################################################################
        predict_class = support_vector_machine_classifier_(train_data, train_class, test_data)
        check_classified_result_(predict_class, test_class, test_index, true_class_num)

    predict_dict = overlap_window_(30, frame_result_dict)
    calculate_evaluation_(predict_dict, ground_truth_dict)

########################################################################
#                      make result movie and graph                     #
########################################################################
"""
    print('make moving picture')
    for key in true_positive_dict.keys():
        check_true_positive_in_frame_(key, true_positive_dict[key],true_sample_dict[key])
        frame_length = frame[files.index(key)]
        visualize_classification_result_(frame_length,true_positive_dict[key].keys(), key)
"""

if __name__ == '__main__':
    main()
