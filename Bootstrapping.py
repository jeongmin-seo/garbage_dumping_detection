import json
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
import cv2
import os
import glob
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import copy


#########################################################################
#files = [1, 12, 30, 31, 43, 67, 69]
#frame = [404, 732, 1619, 1037, 1906, 874, 486]
#Precision:0.8436331  Recall:0.4432278  Accuracy:0.9328258
#########################################################################

#########################################################################
#files = [5,14,19,30,33,38,44,189]
#frame = [703,453,2173,1619,1075,423,1083,287]
#Precision:  Recall:  Accuracy:
#########################################################################

#########################################################################
#files = [5,189]
#frame = [703,287]
#########################################################################

##########################################################################
#files = [11, 12, 15, 17, 18, 28, 31,41, 42, 58, 100, 113, 115, 117,
#         120,124, 125, 127, 133, 136, 140, 144, 147, 153, 160, 161, 164, 165,
#        172, 186, 189, 191, 196, 199, 202, 204, 206,213] #[1, 12, 30, 31, 43, 67, 69]
#frame = [1030, 732, 319, 278, 224, 755, 1037, 871, 1442, 1761, 288, 170, 269, 1049,
#        99, 214, 408, 499, 364, 202, 329, 359, 254, 419, 314, 135, 369, 269,
#         839,628,287,522,715, 1194, 176,744, 1028,252] #[404, 732, 1619, 1037, 1906, 874, 486]
#Precision:0.8785148  Recall:0.3316766  Accuracy:0.8999
#Precision:0.870783  Recall:0.3367419  Accuracy:0.8981
#Precision:0.898605  Recall:0.3662108  Accuracy:0.909791
###########################################################################

###########################################################################
#teahun's file
#files = [15,17,18,28,31,41,42]
#frame = [319, 278, 224, 755, 1037, 871, 1442]
###########################################################################

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
#true_positive_frame = {}
true_sample_dict={}
#true_sample_frame={}

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
        #pose_file = "/home/jmseo/PycharmProjects/ETRIsvm/macrolabeling/macrojson/%03d/%03d_%012d_keypoints.json" % (file_index, file_index, file_number)
        #pose_file = "/home/jmseo/Desktop/ETRI/%03d/%03d_%012d_keypoints.json" % (file_index, file_index, file_number)
        #pose_file = "/home/jmseo/PycharmProjects/ETRIsvm/teahun/%03d/%03d_%012d_keypoints.json" % (file_index, file_index, file_number)
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
#            if pose_list[1]['pose_keypoints'][36] != 0 and pose_list[1]['pose_keypoints'][37] != 0 and \
#                            pose_list[1]['pose_keypoints'][39] != 0 and pose_list[1]['pose_keypoints'][40] != 0:

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
        pose_file = "/home/jmseo/PycharmProjects/ETRIsvm/macrolabeling/macrojson/%03d/%03d_%012d_keypoints.json" \
                    % (file_num, file_num, frame_num)

        # pose_file = "/home/jmseo/PycharmProjects/ETRIsvm/result/%03d/%03d_%012d_keypoints.json" % (file_num, file_num, frame_num)
        #pose_file = "/home/jmseo/PycharmProjects/ETRIsvm/teahun/%03d/%03d_%012d_keypoints.json" % (
        #file_num, file_num, frame_num)
        # pose_file = "/home/jmseo/PycharmProjects/ETRIsvm/throw/%03d/%03d_%012d_keypoints.json" % (file_num, file_num, frame_num)
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

            #posi_list.pop(0)

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

            elif test_class[index] == true_class_label_number:
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

            elif test_class[index] == true_class_label_number:
                false_negative += 1

    print('True Class: %d' % true_class_num,
          'True Positive: %d' % true_positive,
          'True Negative: %d' % true_negative,
          'False Positive: %d' % false_positive,
          'Missing: %d' % false_negative,
          'All: %d' % len(test_class))

    print('Precision: %f' % (float(true_positive)/(true_positive+false_positive)),
          'Recall: %f' % (float(true_positive)/(true_positive+false_negative)),
          'Accuracy: %f' % (float(true_positive + true_negative)/len(test_class)))


def k_means_check_result_(predict_class, test_index, k):

    for i in range(0,k):
        kmeans_dict[i].append([])

    for index in range(0, len(predict_class)):

        if predict_class[index] == 0:
            kmeans_dict[file_info[test_index[index]][0]].append(file_info[test_index[index]][1])

        elif predict_class[index] == 1:
            kmeans_dict[file_info[test_index[index]][0]].append(file_info[test_index[index]][1])



########################################################################
#             predict class using k-means cluster algorithm            #
########################################################################
def kmeans_classifier_(train_data, train_class, test_data):
    from sklearn.cluster import KMeans
    #KMeans(n_clusters=3, random_state=True).fit(train_data)

    return KMeans(n_clusters=3, random_state=True).fit(train_data).predict(test_data)


########################################################################
#             predict class using support vector machine               #
########################################################################
def support_vector_machine_classifier_(train_data, train_class, test_data):
    from sklearn.svm import SVC
    # Support vector machine
    #svc = SVC(kernel='linear', C=0.1).fit(train_data, train_class)
    #y_predict = svc.predict(test_data)

    # if you want weighted class svm, you could adjust class_weight parameter like "class_weight={0:0.08,3:0.92}"
    return SVC(kernel='linear', C=0.1).fit(train_data, train_class).predict(test_data)


def boot_strapping_(train_data, train_class):
    from sklearn.svm import SVC
    """
    iter_number = 0
    while iter_number<5:

        iter_number += 1
    """

    return SVC(kernel='linear', C=0.1).fit(train_data, train_class).decision_function(train_data)


def threshold_false_sample(train_data, train_class, distance):

    mean = sum(distance)/len(distance)
    threshold_data = []
    threshold_class = []
    for i, dist in enumerate(distance):
        if dist > mean:                   #TODO:  should change delete index
            threshold_data.append(train_data[i])
            threshold_class.append(train_class[i])

    return threshold_data, threshold_class

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

    print(len(scaling_key_point))
    true = 0
    for pose in pose_class:
        if pose == true_class_label_number:
            true += 1
    print(true)
    print("k-fold")

    training_data = copy.deepcopy(coord_key_point)
    training_class = copy.deepcopy(pose_class)
    for i in range(0, 4):
        distance = boot_strapping_(training_data, training_class)
        print(min(distance), max(distance))
        (training_data, training_class) = threshold_false_sample(training_data, training_class, distance)
        filt = list(filter(lambda x: x < 1, training_class))
        print(len(training_data), len(filt),len(training_data)-len(filt))
        print("-----------------------")

    svm = SVC(kernel='linear', C=0.1).fit(training_data, training_class)
########################################################################
#       10-fold validation and split training & test data set          #
########################################################################
    """
    # 10fold & shuffle = True
    skf = StratifiedKFold(n_splits=1, shuffle=True)

    # split train test
    for train_index, test_index in skf.split(coord_key_point, pose_class):
        train_data = []
        test_data = []
        train_class = []
        test_class = []
        true_class_num = 0
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

        for i in range(0,6):
            distance = boot_strapping_(train_data,train_class)
            print(min(distance), max(distance))
            (train_data, train_class) = threshold_false_sample(train_data, train_class, distance)
        print("-----------------------")

        #predict_class = support_vector_machine_classifier_(train_data, train_class, test_data)
        predict_class = svm.predict(test_data)
        check_classified_result_(predict_class, test_class, test_index, true_class_num)
        """
    predict_class = svm.predict(coord_key_point)
    check_classified_result_(predict_class, pose_class, 0, 123)

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
