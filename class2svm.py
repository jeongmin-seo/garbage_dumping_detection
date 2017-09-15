import json
from sklearn.model_selection import StratifiedKFold
import cv2
import os
import glob
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import random
import copy



###########################################################################
#class 2
files =[6, 13, 23, 46, 57, 61, 65, 69, 72, 95, 99]
frame = [593, 329, 509, 639, 824, 134, 457, 486, 179, 319, 599]
###########################################################################

###########################################################################
#class 3
#files =[123,157,183,184,187,188,192,195,205,214]
#frame = [219, 409, 905, 1596, 556, 367, 279, 271, 340, 344]
###########################################################################


true_class_label_number = 2 #change true class label number
file_info = []
kmeans_dict={}
true_positive_dict = {}
true_sample_dict={}

for i in files:
    true_positive_dict[i] = {}
    true_sample_dict[i] = {}


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

    while file_number < end_num:
        #class2
        pose_file = "/home/jmseo/PycharmProjects/ETRIsvm/class2json/%03d/%03d_%012d_keypoints.json" % (file_index, file_index, file_number)
        #class3
        #pose_file = "/home/jmseo/PycharmProjects/ETRIsvm/class3json/%03d/%03d_%012d_keypoints.json" % (file_index, file_index, file_number)

        dict_pose = read_pose_(pose_file)  #dict_pose = {}
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
                    if not file_number in true_sample_dict[file_index]:
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
def check_true_positive_in_frame_(file_num, posi_dict,true_dict):
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
        pose_file = "/home/jmseo/PycharmProjects/ETRIsvm/class2json/%03d/%03d_%012d_keypoints.json" % (file_num, file_num, frame_num)
        dict_pose = read_pose_(pose_file)  # dict_pose={}
        people = dict_pose['people']

        if frame_num in true_dict:
            for i in true_dict[frame_num]:
                temp = people[i]['pose_keypoints']
                temp_x = [temp[x * 3] for x in range(0, 18)]
                temp_y = [temp[y * 3 + 1] for y in range(0, 18)]
                cv2.rectangle(frame,
                              (
                              int(min(filter(lambda x: x > 0, temp_x))), int(min(filter(lambda x: x > 0, temp_y)))),
                              (int(max(temp_x)), int(max(temp_y))),
                              (255, 0, 0), thickness=3)

        if posi_dict and frame_num in posi_dict:
            cv2.circle(frame, (int(width)-200, int(height)-100), 40, (255, 255, 255), 40)
            """
            # add new code
            pose_file = "/home/jmseo/PycharmProjects/ETRIsvm/macrolabeling/macrojson/%03d/%03d_%012d_keypoints.json" \
                       % (file_num, file_num, frame_num)

            #pose_file = "/home/jmseo/PycharmProjects/ETRIsvm/class2json/%03d/%03d_%012d_keypoints.json" % (file_index, file_index, file_number)
            pose_file = "/home/jmseo/PycharmProjects/ETRIsvm/teahun/%03d/%03d_%012d_keypoints.json" % (file_num, file_num, frame_num)
            #pose_file = "/home/jmseo/PycharmProjects/ETRIsvm/throw/%03d/%03d_%012d_keypoints.json" % (file_num, file_num, frame_num)
            dict_pose = read_pose_(pose_file)  #dict_pose={}
            people = dict_pose['people']
            """
            for i in posi_dict[frame_num]:
                temp = people[i]['pose_keypoints']
                temp_x = [temp[x*3] for x in range(0, 18)]
                temp_y = [temp[y*3 + 1] for y in range(0, 18)]
                cv2.rectangle(frame,
                              (int(min(filter(lambda x: x > 0, temp_x))+5), int(min(filter(lambda x: x > 0, temp_y))+5)),
                              (int(max(temp_x)+5), int(max(temp_y)+5)),
                              (0, 0, 255), thickness=3)


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

        #change true class label
        if predict_class[index] == true_class_label_number:
            if test_class[index] == 0:
                false_positive += 1

            else:
                true_positive += 1

        elif predict_class[index] == 0:
            if test_class[index] == 0:
                true_negative += 1

            else: #elif test_class[index] == true_class_label_number:
                false_negative += 1

    print('True Class: %d' % true_class_num,
          'True Positive: %d' % true_positive,
          'True Negative: %d' % true_negative,
          'False Positive: %d' % false_positive,
          'Missing: %d' % false_negative,
          'All: %d' % len(test_class))
"""
    print('Precision: %f' % (float(true_positive)/(true_positive+false_positive)),
          'Recall: %f' % (float(true_positive)/(true_positive+false_negative)),
          'Accuracy: %f' % (float(true_positive + true_negative)/len(test_class)))
"""


########################################################################
#             predict class using support vector machine               #
########################################################################
def support_vector_machine_classifier_(train_data, train_class, test_data):
    from sklearn.svm import SVC
    # Support vector machine
    #svc = SVC(kernel='linear', C=0.1).fit(train_data, train_class)
    #y_predict = svc.predict(test_data)

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

    plt.title('%d file' %file_number)
    plt.savefig('%d file' %file_number)



def main():
    print('start')
    key_point = []
    pose_class = []

########################################################################
#                     import data to python list                       #
########################################################################
    for i in range(0, len(files)):
        tmp_key_point, tmp_pose_class, tmp_positive, tmp_negative = import_data_(0, frame[i], i)
        if i == 0:
            key_point = tmp_key_point
            pose_class = tmp_pose_class
            if not true_class_label_number == 1:
                positive = tmp_positive
                negative = tmp_negative

        else:
            key_point.extend(tmp_key_point)
            pose_class.extend(tmp_pose_class)
            if not true_class_label_number == 1:
                positive.extend(tmp_positive)
                negative.extend(tmp_negative)

########################################################################
#                      normalize & scaling data                        #
########################################################################

    if true_class_label_number == 1:
        norm_key_point = normalize_pose_(key_point)
        scaling_key_point = scaling_data_(norm_key_point)

    else:
        norm_positive_keypoint = normalize_pose_(positive)
        norm_negative_keypoint = normalize_pose_(negative)
        scaling_positive_keypoint = scaling_data_(norm_positive_keypoint)
        scaling_negative_keypoint = scaling_data_(norm_negative_keypoint)
        positive_class = [true_class_label_number] * len(scaling_positive_keypoint)
        negative_class = [0] * len(scaling_negative_keypoint)


########################################################################
#               extract body coordinate except confidence              #
########################################################################

    coord_positive_keypoint = []
    for point in scaling_positive_keypoint:
        coord_positive_keypoint.append([])
        for index in range(0,18):
            coord_positive_keypoint[len(coord_positive_keypoint) - 1].append(point[index * 3])
            coord_positive_keypoint[len(coord_positive_keypoint) - 1].append(point[index * 3 + 1])

    coord_negative_keypoint = []
    for point in scaling_negative_keypoint:
        coord_negative_keypoint.append([])
        for index in range(0, 18):
            coord_negative_keypoint[len(coord_negative_keypoint) - 1].append(point[index * 3])
            coord_negative_keypoint[len(coord_negative_keypoint) - 1].append(point[index * 3 + 1])


    test_data = copy.deepcopy(coord_negative_keypoint)
    test_data.extend(coord_positive_keypoint)
    test_class = copy.deepcopy(negative_class)
    test_class.extend(positive_class)


########################################################################
#       10-fold validation and split training & test data set          #
########################################################################
    print("all nega",len(coord_negative_keypoint))
    print("all posi",len(coord_positive_keypoint))
    random.shuffle(coord_negative_keypoint)
    for i in range(0,10):
        num = int(len(coord_negative_keypoint)/10)
        train_data = coord_negative_keypoint[num*i:num*(i+1)]
        print(len(train_data))
        train_class = [0]*len(train_data)
        train_data.extend(coord_positive_keypoint)
        print(len(coord_positive_keypoint))
        train_class.extend(positive_class)



########################################################################
#     predict test class & calculate Precision Recall and Accuracy     #
########################################################################

        predict_class = support_vector_machine_classifier_(train_data, train_class, test_data)
        check_classified_result_(predict_class, test_class, 0, len(positive_class))


if __name__ == '__main__':
    main()
