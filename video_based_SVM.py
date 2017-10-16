import json
import cv2
import matplotlib.pyplot as plt
import numpy as np
from xml.etree.ElementTree import parse
from sklearn.model_selection import KFold

files = [5, 15, 17, 18, 28,
         31, 41, 42, 58, 100,
         113, 115, 117, 120, 124,
         125, 127, 133, 136, 140,
         144, 147, 153, 160, 161,
         164, 165, 172, 186, 191,
         196, 199, 202, 206, 213
         ]

frame = [703, 319, 278, 224, 755,
         1037, 871, 1442, 1761, 288,
         170, 269, 1049, 99, 214,
         408, 499, 364, 202, 329,
         359, 254, 419, 314, 135,
         369, 269, 839, 628, 522,
         715, 1194, 176, 1028, 252]

true_class_label_number = 1  # change true class label number

# graph related
true_sample_dict = {}
true_positive_dict = {}
predict_positive_dict = {}

# evaluation related
frame_result_dict = {}
ground_truth_dict = {}
gt_and_detect_result = {}


########################################################################
#                         file import related                          #
########################################################################

# read json file
def read_pose_(filename):
    f = open(filename, 'r')
    js = json.loads(f.read())
    f.close()

    return js


#
def import_train_data_(start_num, end_num, index):
    pose_key_points = []
    y = []
    negative_sample = []
    positive_sample = []
    file_number = start_num
    file_index = files[index]

    while file_number <= end_num:
        pose_file = "C:\Users\JM\Desktop\Data\ETRIrelated\jsonfile_class%d\%03d\%03d_%012d_keypoints.json" \
                    % (true_class_label_number, file_index, file_index, file_number)

        dict_pose = read_pose_(pose_file)  # dict_pose = {}
        people = dict_pose['people']

        if not people:
            file_number += 1
            continue

        for pose_list in enumerate(people):
            if pose_list[1]['pose_keypoints'][29] != 0 and pose_list[1]['pose_keypoints'][32] != 0 and \
                            pose_list[1]['pose_keypoints'][38] != 0 and pose_list[1]['pose_keypoints'][41] != 0 and \
                            pose_list[1]['pose_keypoints'][5] != 0:

                if len(pose_list[1]['pose_keypoints']) == 54:
                    negative_sample.append(pose_list[1]['pose_keypoints'])
                    y.append(0)

                elif len(pose_list[1]['pose_keypoints']) == 55:
                    positive_sample.append(pose_list[1]['pose_keypoints'])
                    y.append(pose_list[1]['pose_keypoints'].pop())

                else:
                    print(pose_file)

                pose_key_points.append(pose_list[1]['pose_keypoints'])
        file_number += 1

    return pose_key_points, y, positive_sample, negative_sample


def import_test_data_(start_num, end_num, index):
    pose_key_points = []
    y = []
    negative_sample = []
    positive_sample = []
    file_number = start_num
    file_index = files[index]

    while file_number <= end_num:
        pose_file = "C:\Users\JM\Desktop\Data\ETRIrelated\jsonfile_class%d\%03d\%03d_%012d_keypoints.json" \
                    % (true_class_label_number, file_index, file_index, file_number)

        frame_result_dict[file_index].append(False)  # evaluation related

        dict_pose = read_pose_(pose_file)  # dict_pose = {}
        people = dict_pose['people']

        if not people:
            file_number += 1
            continue

        for pose_list in enumerate(people):
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


# make ground truth using text file
def read_ground_truth_using_text(filename):
    f = open(filename, 'r')
    lines = f.readlines()

    for line in lines:
        split_line = line.split(" ")
        file_number = int(split_line[0])
        start = int(split_line[1])
        end = int(split_line[2])
        frame_length = frame[files.index(file_number)] + 1

        ground_truth_dict[file_number] = [False] * frame_length
        for index in range(start, end + 1):
            ground_truth_dict[file_number][index] = True

    f.close()


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
            pose[base_index * 3] -= neck_x
            pose[base_index * 3 + 1] -= neck_y
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
            pose[base_index * 3] /= dist
            pose[base_index * 3 + 1] /= dist
            base_index += 1

    return scaling_pose_data


def make_normal_vector(pose_data):
    normal_vector = [pose for pose in pose_data]

    for pose in normal_vector:
        for base_index in range(18):
            dist = (pose[base_index*3] ** 2 + pose[base_index*3+1] ** 2) ** 0.5

            if dist == 0:
                continue

            pose[base_index*3] /= dist
            pose[base_index*3 + 1] /= dist

    return normal_vector


########################################################################
#                 make the true positive result movie                  #
########################################################################
def check_true_positive_in_frame_(file_num, _gt_dict, _tp_dict, _window_list):
    file_path = "D:/etri_tool/CPMresult/videoresult/%03dresult.avi" % file_num

    cap = cv2.VideoCapture(file_path)
    fps, width, height = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(
        cv2.CAP_PROP_FRAME_HEIGHT)
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
        pose_file = "C:\Users\JM\Desktop\Data\ETRIrelated\jsonfile_class%d\%03d\%03d_%012d_keypoints.json" \
                    % (true_class_label_number, file_num, file_num, frame_num)

        dict_pose = read_pose_(pose_file)  # dict_pose={}
        people = dict_pose['people']

        if frame_num in _gt_dict.keys():
            for i in _gt_dict[frame_num]:
                temp = people[i]['pose_keypoints']
                temp_x = [temp[x * 3] for x in range(0, 18)]
                temp_y = [temp[y * 3 + 1] for y in range(0, 18)]
                cv2.rectangle(frame,
                              (int(min(filter(lambda x: x > 0, temp_x))), int(min(filter(lambda x: x > 0, temp_y)))),
                              (int(max(temp_x)), int(max(temp_y))),
                              (255, 0, 0), thickness=3)

        if frame_num in _tp_dict.keys():
            for i in _tp_dict[frame_num]:
                temp = people[i]['pose_keypoints']
                temp_x = [temp[x * 3] for x in range(0, 18)]
                temp_y = [temp[y * 3 + 1] for y in range(0, 18)]
                cv2.rectangle(frame,
                              (int(min(filter(lambda x: x > 0, temp_x)))-5, int(min(filter(lambda x: x > 0, temp_y)))-5),
                              (int(max(temp_x))+5, int(max(temp_y))+5),
                              (0, 0, 255), thickness=3)

        if _window_list[frame_num]:
            cv2.circle(frame, (int(width) - 200, int(height) - 100), 40, (255, 255, 255), 40)

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
def check_classified_result_(predict_class, test_class, test_index):
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

            if frame_idx not in predict_positive_dict[video_idx]:
                predict_positive_dict[video_idx][frame_idx] = []
            predict_positive_dict[video_idx][frame_idx].append(file_info[test_index[index]][2])

            if test_class[index] == 0:
                false_positive += 1

            else:
                true_positive += 1

        elif predict_class[index] == 0:
            if test_class[index] == 0:
                true_negative += 1

            else:  # elif test_class[index] == true_class_label_number:
                false_negative += 1

    print("precision: ", float(true_positive) / float(true_positive + false_positive))
    print("recall: ", float(true_positive) / float(true_positive + false_negative))
    print("accuracy: ",
          float(true_positive + true_negative) / float(true_positive + false_positive + false_negative + true_negative))


def check_classified_result_using_video(predict_class, test_class):
    false_positive = 0
    false_negative = 0
    true_positive = 0
    true_negative = 0

    for index in range(len(file_info)):
        if predict_class[index] == true_class_label_number:

            video_idx = file_info[index][0]
            frame_idx = file_info[index][1]

            # Draw graph related
            frame_result_dict[video_idx][frame_idx] = True

            # Save predict positive result
            if frame_idx not in predict_positive_dict[video_idx]:
                predict_positive_dict[video_idx][frame_idx] = []
            predict_positive_dict[video_idx][frame_idx].append(file_info[index][2])

            if test_class[index] == 0:
                false_positive += 1

            else:
                true_positive += 1

        elif predict_class[index] == 0:
            if test_class[index] == 0:
                true_negative += 1

            else:  # elif test_class[index] == true_class_label_number:
                false_negative += 1
    """
    precision = float(true_positive) / float(true_positive + false_positive)
    recall = float(true_positive) / float(true_positive + false_negative)
    accuracy = float(true_positive + true_negative) / \
               float(true_positive + false_positive + false_negative + true_negative)

    print("precision: ", precision)
    print("recall: ", recall)
    print("accuracy: ", accuracy)
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
def visualize_classification_result_(frame_length, _ground_truth_list, file_number, _predict_list, _none_window_list):
    fig = plt.figure()
    ax = fig.add_subplot(313)
    bx = fig.add_subplot(312)
    cx = fig.add_subplot(311)

    plt.suptitle('%d file' % file_number).set_position((30, 30))

    frame_length = frame_length + 1
    xs = np.arange(frame_length)
    ys = [0] * frame_length

    # You can provide either a single color or an array. To demonstrate this,
    # the first bar of each set will be colored cyan.
    cs = [0] * frame_length
    ds = [0] * frame_length

    for i in range(0, frame_length):
        if _none_window_list[i]:
            ds[i] += 1

        if _ground_truth_list[i]:
            cs[i] += 1

        if _predict_list[i]:
            ys[i] += 1

    ax.plot(xs, cs, color='darkred')
    ax.set_title(" Ground Truth ")

    bx.plot(xs, ds, color='darkred')
    bx.set_title(" Prediction ")

    """
    num = 0
    for _list in _predict_list:
        for cr in range(_list[0], _list[1] + 1):
            ys[cr] += 1
            num += 1
            if num == 1:
                cx.text(50, 30, i)
            continue
    """
    cx.plot(xs, ys, color='darkred')
    cx.set_title(" Prediction overlap Window ")

    ax.set_xlabel('time')

    plt.tight_layout()

    plt.savefig('%d file' % file_number)
    plt.close(fig)


########################################################################
#                     Calc Overlap Window related                      #
########################################################################

# cover overlap window
def overlap_window_all_cover(window_size_, predict_result_):
    result_dict = {}
    result_range_dict = {}
    for key in predict_result_.keys():
        size = len(predict_result_[key])
        result_dict[key] = [False] * size
        result_range_dict[key] = []

        index = 0
        while index < size:
            if predict_result_[key][index] == true_class_label_number:
                start = max(0, index - window_size_)
                end = min(size, index + window_size_)

                for i in range(start, end):
                    result_dict[key][i] = True
                index = end
                continue

            index += 1

    return result_dict


# calculate positive sample frame range
def check_positive_range(_dictionary):
    result = {}
    cur_state = False
    start = 0
    end = 0

    for key in _dictionary.keys():

        result[key] = []
        for size in range(0, frame[files.index(key)]):

            if cur_state != _dictionary[key][size]:
                if cur_state:
                    cur_state = False
                    end = size
                    result[key].append([start, end])

                else:
                    cur_state = True
                    start = size

    return result


# detect criteria calc result
def detect_base_calculate_result_(_detect_range_list, _gt_and_detect_list):
    false_posi = 0
    for detect_range in _detect_range_list:

        if not detect_range:
            continue

        denominator = float(detect_range[1] - detect_range[0] + 1)
        rate = float(_gt_and_detect_list[detect_range[0]: detect_range[1]].count(True)) / denominator
        if rate < 0.5:
            false_posi += 1

    return false_posi


# ground truth criteria calc result
def gt_base_calculate_result_(_gt_range_list, _gt_and_detect_list):
    true_posi = 0
    false_nega = 0
    for gt_range in _gt_range_list:

        if not gt_range:
            continue

        denominator = float(gt_range[1] - gt_range[0] + 1)
        rate = float(_gt_and_detect_list[gt_range[0]: gt_range[1]].count(True)) / denominator

        if rate >= 0.5:
            true_posi += 1
        else:
            false_nega += 1

    return true_posi, false_nega


# calc recall & precision
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

    recall = float(true_posi) / float(true_posi + false_nega)
    precision = float(true_posi) / float(true_posi + false_posi)

    print ("Recall:", recall)
    print ("Precision:", precision)

    return _detect_result_range


def initialize_result_container_():
    true_sample_dict.clear()
    true_positive_dict.clear()
    predict_positive_dict.clear()
    frame_result_dict.clear()
    ground_truth_dict.clear()

if __name__ == '__main__':

    print('start')

    X = np.array(range(len(files)))
    kf = KFold(n_splits=10, random_state=True, shuffle=True)
    for train_index, test_index in kf.split(X):

        file_info = []
        initialize_result_container_()

        test_list = test_index.tolist()
        train_list = train_index.tolist()
        for i in test_list:
            print(files[i])
            true_positive_dict[files[i]] = {}
            predict_positive_dict[files[i]] = {}
            true_sample_dict[files[i]] = {}
            frame_result_dict[files[i]] = []
            ground_truth_dict[files[i]] = []
        read_ground_truth_using_text("C:\Users\JM\Desktop\Data\ETRIrelated\pose classification\class1macro.txt")

        test_keypoint = []
        test_class = []
        train_keypoint = []
        train_class = []

        # train data import
        for idx in train_list:
            tmp_key_point, tmp_pose_class, tmp_positive, tmp_negative = import_train_data_(0, frame[idx], idx)
            if train_list.index(idx) == 0:
                train_keypoint = tmp_key_point
                train_class = tmp_pose_class

            else:
                train_keypoint.extend(tmp_key_point)
                train_class.extend(tmp_pose_class)

        # test data import
        for idx in test_list:
            tmp_key_point, tmp_pose_class, tmp_positive, tmp_negative = import_test_data_(0, frame[idx], idx)
            if test_list.index(idx) == 0:
                test_keypoint = tmp_key_point
                test_class = tmp_pose_class

            else:
                test_keypoint.extend(tmp_key_point)
                test_class.extend(tmp_pose_class)

        ########################################################################
        #                      normalize & scaling data                        #
        ########################################################################
        norm_train_keypoint = normalize_pose_(train_keypoint)
        norm_test_keypoint = normalize_pose_(test_keypoint)
        scaling_train_keypoint = make_normal_vector(norm_train_keypoint)
        scaling_test_keypoint = make_normal_vector(norm_test_keypoint)

        # scaling_train_keypoint = scaling_data_(norm_train_keypoint)
        # scaling_test_keypoint = scaling_data_(norm_test_keypoint)

        ########################################################################
        #               extract body coordinate except confidence              #
        ########################################################################
        coord_train_keypoint = []
        for point in scaling_train_keypoint:
            coord_train_keypoint.append([])
            for index in range(0, 18):
                coord_train_keypoint[len(coord_train_keypoint) - 1].append(point[index * 3])
                coord_train_keypoint[len(coord_train_keypoint) - 1].append(point[index * 3 + 1])

        coord_test_keypoint = []
        for point in scaling_test_keypoint:
            coord_test_keypoint.append([])
            for index in range(0, 18):
                coord_test_keypoint[len(coord_test_keypoint) - 1].append(point[index * 3])
                coord_test_keypoint[len(coord_test_keypoint) - 1].append(point[index * 3 + 1])

        ########################################################################
        #     predict test class & calculate Precision Recall and Accuracy     #
        ########################################################################

        predict_class = support_vector_machine_classifier_(coord_train_keypoint, train_class,
                                                           coord_test_keypoint)
        check_classified_result_using_video(predict_class, test_class)

        predict_dict = overlap_window_all_cover(30, frame_result_dict)
        predict_range = calculate_evaluation_(predict_dict, ground_truth_dict)

        ########################################################################
        #                      make result movie and graph                     #
        ########################################################################

        print('make moving picture')
        # print(predict_positive_dict.keys())
        for key in predict_positive_dict.keys():
            frame_length = frame[files.index(key)]
            visualize_classification_result_(frame_length, ground_truth_dict[key],
                                             key,
                                             predict_dict[key],
                                             frame_result_dict[key])
            check_true_positive_in_frame_(key, true_sample_dict[key], predict_positive_dict[key], predict_dict[key])


