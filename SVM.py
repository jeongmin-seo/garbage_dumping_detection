import json
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
import cv2
from scipy.optimize import linear_sum_assignment


files = [1, 12, 30, 31, 43, 67, 69]
frame = [404, 732, 1619, 1037, 1906, 874, 486]
file_info = []
true_positive_dict = {}
true_posi_frame = {}
for i in files:
    true_positive_dict[i] = []


def read_pose_(filename):
    f = open(filename, 'r')
    js = json.loads(f.read())
    f.close()

    return js


def import_data_(startnum, endnum, index):
    pose_keypoints = []
    y = []
    filenumber = startnum
    fileindex = files[index]

    while filenumber < endnum:
        POSE_FILE = "/home/jmseo/Desktop/ETRI/%03d/%03d_%012d_keypoints.json" % (fileindex, fileindex, filenumber)
        POSE = {}
        POSE = read_pose_(POSE_FILE)
        people = POSE['people']

        if not people:
            filenumber += 1
            continue

        for pose_list in enumerate(people):
            if pose_list[1]['pose_keypoints'][36] != 0 and pose_list[1]['pose_keypoints'][37] != 0 and \
                            pose_list[1]['pose_keypoints'][39] != 0 and pose_list[1]['pose_keypoints'][40] != 0:

                if len(pose_list[1]['pose_keypoints']) == 54:
                    y.append(0)

                elif len(pose_list[1]['pose_keypoints']) == 55:
                    y.append(pose_list[1]['pose_keypoints'].pop())

                else:
                    print(POSE_FILE)

                pose_keypoints.append(pose_list[1]['pose_keypoints'])
                file_info.append([fileindex, filenumber, pose_list[0]])

        filenumber += 1
    return pose_keypoints, y


def normalize_pose_(posedata):
    norm_posedata = [pose for pose in posedata]

    for pose in norm_posedata:
        neck_x = pose[3]
        neck_y = pose[4]
        baseindex = 0

        while baseindex < 18:
            pose[baseindex*3] -= neck_x
            pose[baseindex*3+1] -= neck_y
            baseindex += 1

    return norm_posedata


def scaling_data_(posedata):
    scaling_posedata = [pose for pose in posedata]

    for pose in scaling_posedata:
        Lknee, LAnkle = [pose[36], pose[37]], [pose[39], pose[40]]
        Rknee, RAnkle = [pose[27], pose[28]], [pose[30], pose[31]]
        baseindex = 0

        Rdist = ((Rknee[0] - RAnkle[0]) ** 2 + (Rknee[1] - RAnkle[1]) ** 2) ** 0.5
        Ldist = ((Lknee[0] - LAnkle[0]) ** 2 + (Lknee[1] - LAnkle[1]) ** 2) ** 0.5
        dist = Rdist if Rdist > Ldist else Ldist

        while baseindex < 18:
            pose[baseindex*3] /= dist
            pose[baseindex*3+1] /= dist
            baseindex += 1

    return scaling_posedata


def check_true_positive_in_frame_(file_num, posi_list):
    file_path = "/home/jmseo/openpose/examples/media/ETRI/%03dresult.avi" % file_num

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

        if posi_list and frame_num == posi_list[0]:
            cv2.circle(frame, (int(width)-200, int(height)-100), 40, (255, 255, 255), 40)

            # add new code
            POSE_FILE = "/home/jmseo/Desktop/ETRI/%03d/%03d_%012d_keypoints.json" % (file_num, file_num, posi_list[0])
            POSE = {}
            POSE = read_pose_(POSE_FILE)
            people = POSE['people']

            for i in true_posi_frame[posi_list[0]]:
                temp = people[i]['pose_keypoints']
                temp_x = [temp[i*3] for i in range(0, 18)]
                temp_y = [temp[i*3 + 1] for i in range(0, 18)]
                cv2.rectangle(frame,
                              (int(min(filter(lambda x: x > 0, temp_x))), int(min(filter(lambda x: x > 0, temp_y)))),
                              (int(max(temp_x)), int(max(temp_y))),
                              (0, 0, 255), thickness=3)

            posi_list.pop(0)

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


def main():
    keypoint = []
    pose_class = []

    for i in range(0, len(files)):
        tmp_keypoint, tmp_pose_class = import_data_(0, frame[i], i)
        if i == 0:
            keypoint = tmp_keypoint
            pose_class = tmp_pose_class

        else:
            keypoint.extend(tmp_keypoint)
            pose_class.extend(tmp_pose_class)

    norm_keypoint = normalize_pose_(keypoint)
    scaling_keypoint = scaling_data_(norm_keypoint)

    coord_keypoint = []
    for point in scaling_keypoint:
        coord_keypoint.append([])
        for i in range(0, 18):
            coord_keypoint[len(coord_keypoint) - 1].append(point[i * 3])
            coord_keypoint[len(coord_keypoint) - 1].append(point[i * 3 + 1])

    # 10fold & shuffle = True
    skf = StratifiedKFold(n_splits=10, shuffle=True)

    # split train test
    for train_index, test_index in skf.split(coord_keypoint, pose_class):
        train_data = []
        test_data = []
        train_class = []
        test_class = []
        class_num = 0
        for index in train_index:
            train_data.append(coord_keypoint[index])
            train_class.append(pose_class[index])
        for index in test_index:
            test_data.append(coord_keypoint[index])
            test_class.append(pose_class[index])
            if pose_class[index] == 1:
                class_num += 1

        # Support vector machine
        svc = SVC(kernel='linear', C=0.1).fit(train_data, train_class)
        y_pred = svc.predict(test_data)

        false_posi = 0
        false_nega = 0
        true_posi = 0
        true_nega = 0
        for i in range(0, len(y_pred)):

            if y_pred[i] == 1:
                if test_class[i] == 1:
                    true_posi += 1
                    true_positive_dict[file_info[test_index[i]][0]].append(file_info[test_index[i]][1])

                    if not file_info[test_index[i]][1] in true_posi_frame:
                        true_posi_frame[file_info[test_index[i]][1]] = []

                    true_posi_frame[file_info[test_index[i]][1]].append(file_info[test_index[i]][2])

                elif test_class[i] == 0:
                    false_posi += 1

            elif y_pred[i] == 0:
                if test_class[i] == 0:
                    true_nega += 1

                elif test_class[i] == 1:
                    false_nega += 1

        print('True Class: %d' % class_num,
              'True Positive: %d' % true_posi,
              'True Negative: %d' % true_nega,
              'False Positive: %d' % false_posi,
              'Missing: %d' % false_nega,
              'All: %d' % len(test_class))

    for i in true_positive_dict.keys():
        true_positive_dict[i].sort()
        check_true_positive_in_frame_(i, true_positive_dict[i])


if __name__ == '__main__':
    main()
