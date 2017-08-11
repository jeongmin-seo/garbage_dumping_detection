import json
from sklearn.model_selection import StratifiedKFold
import cv2
import os
import glob


#dumping_pose_file_num = range(73,104)

#files = [1, 12, 30, 31, 43, 67, 69]
#frame = [404, 732, 1619, 1037, 1906, 874, 486]

files = [5,14,19,30,33,38,44]
frame = [703,453,2173,1619,1075,423,1083]

#files = [11, 12, 15, 17, 18, 28, 31,41, 42, 58, 100, 113, 115, 117,
#         120, 127, 133, 136, 140, 144, 147, 153, 160, 161, 164, 165] #[1, 12, 30, 31, 43, 67, 69]
#frame = [1030, 732, 319, 278, 224, 755, 1037, 871, 1442, 1761, 288, 170, 269, 1049,
#         99, 499, 364, 202, 329, 359, 254, 419, 314, 135, 369, 269] #[404, 732, 1619, 1037, 1906, 874, 486]
file_info = []
kmeans_dict={}
true_positive_dict = {}
true_positive_frame = {}
for i in files:
    true_positive_dict[i] = {}


def read_pose_(filename):
    f = open(filename, 'r')
    js = json.loads(f.read())
    f.close()

    return js


def import_data_(start_num, end_num, index):
    pose_key_points = []
    y = []
    file_number = start_num
    file_index = files[index]

    while file_number < end_num:
        #pose_file = "/home/jmseo/PycharmProjects/ETRIsvm/result/%03d/%03d_%012d_keypoints.json" % (file_index, file_index, file_number)
        pose_file = "/home/jmseo/Desktop/ETRI/%03d/%03d_%012d_keypoints.json" % (file_index, file_index, file_number)
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
                    y.append(0)

                elif len(pose_list[1]['pose_keypoints']) == 55:
                    y.append(pose_list[1]['pose_keypoints'].pop())

                else:
                    print(pose_file)

                pose_key_points.append(pose_list[1]['pose_keypoints'])
                file_info.append([file_index, file_number, pose_list[0]])

        file_number += 1

    return pose_key_points, y


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


def check_true_positive_in_frame_(file_num, posi_dict):
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

        if posi_dict and frame_num in posi_dict:
            cv2.circle(frame, (int(width)-200, int(height)-100), 40, (255, 255, 255), 40)

            # add new code
            pose_file = "/home/jmseo/Desktop/ETRI/%03d/%03d_%012d_keypoints.json" % (file_num, file_num, frame_num)
            dict_pose = read_pose_(pose_file)  #dict_pose={}
            people = dict_pose['people']

            for i in posi_dict[frame_num]:
                temp = people[i]['pose_keypoints']
                temp_x = [temp[x*3] for x in range(0, 18)]
                temp_y = [temp[y*3 + 1] for y in range(0, 18)]
                cv2.rectangle(frame,
                              (int(min(filter(lambda x: x > 0, temp_x))), int(min(filter(lambda x: x > 0, temp_y)))),
                              (int(max(temp_x)), int(max(temp_y))),
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


def check_classified_result_(predict_class, test_class, test_index, true_class_num):
    false_positive = 0
    false_negative = 0
    true_positive = 0
    true_negative = 0
    for index in range(0, len(predict_class)):

        if predict_class[index] == 1:
            if test_class[index] == 1:
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

        elif predict_class[index] == 0:
            if test_class[index] == 0:
                true_negative += 1

            elif test_class[index] == 1:
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


def kmeans_classifier_(train_data, train_class, test_data):
    from sklearn.cluster import KMeans
    #KMeans(n_clusters=3, random_state=True).fit(train_data)

    return KMeans(n_clusters=3, random_state=True).fit(train_data).predict(test_data)

def support_vector_machine_classifier_(train_data, train_class, test_data):
    from sklearn.svm import SVC
    # Support vector machine
    #svc = SVC(kernel='linear', C=0.1).fit(train_data, train_class)
    #y_predict = svc.predict(test_data)

    return SVC(kernel='linear', C=0.1).fit(train_data, train_class).predict(test_data)


def main():
    print('start')
    key_point = []
    pose_class = []

    for i in range(0, len(files)):
        tmp_key_point, tmp_pose_class = import_data_(0, frame[i], i)
        if i == 0:
            key_point = tmp_key_point
            pose_class = tmp_pose_class

        else:
            key_point.extend(tmp_key_point)
            pose_class.extend(tmp_pose_class)

    norm_key_point = normalize_pose_(key_point)
    scaling_key_point = scaling_data_(norm_key_point)

    coord_key_point = []
    for point in scaling_key_point:
        coord_key_point.append([])
        for index in range(0, 18):
            coord_key_point[len(coord_key_point) - 1].append(point[index * 3])
            coord_key_point[len(coord_key_point) - 1].append(point[index * 3 + 1])

    print(len(scaling_key_point))
    true = 0
    for pose in pose_class:
        if pose == 1:
            true +=1
    print(true)
    print("k-fold")
    # 10fold & shuffle = True
    skf = StratifiedKFold(n_splits=10, shuffle=True)

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
            if pose_class[index] == 1:
                true_class_num += 1

        predict_class = support_vector_machine_classifier_(train_data, train_class, test_data)
        check_classified_result_(predict_class, test_class, test_index, true_class_num)


    print('make moving picture')
    for key in true_positive_dict.keys():
        check_true_positive_in_frame_(key, true_positive_dict[key])


if __name__ == '__main__':
    main()
