import cv2
import os
import progressbar

limbs = [[1, 2], [1, 5], [2, 3], [3, 4], [5, 6],
         [6, 7], [1, 8], [8, 9], [9, 10], [1, 11],
         [11, 12], [12, 13], [1, 0], [0, 14], [14, 16],
         [0, 15], [15, 17]]

kBasePath = "C:\\Users\\JM\\Desktop\\Data\\ETRIrelated\\BMVC\\"
kResultVideoBasePath = os.path.join(kBasePath, "point_check")
kKeyPointBasePath = os.path.join(kBasePath, "processed")

kKeyPointPath = {'original': os.path.join(kBasePath, "pose_data_confidence"),
                 'smoothed': os.path.join(kBasePath, "processed")}
kVideoName = {'original': "%03d.avi",
              'smoothed': "%03d_point_smoothing.avi"}
kPointColor = {'original': (0, 255, 0),
               'smoothed': (0, 0, 255)}


def read_keypoint(_file_path):
    point = {}

    f = open(_file_path,'r')
    for line in f.readlines():
        split_line = line.split(' ')
        person_id = int(split_line[0])
        frame_num = int(split_line[2])

        if frame_num not in point.keys():
            point[frame_num] = {}

        point[frame_num][person_id] = split_line[3:-1]

    f.close()

    return point


def draw_keypoint(input_frame, _point, _color, frame_index):

    if frame_index in _point.keys():
        frame_all_point = _point[frame_index]

        for p_id in frame_all_point.keys():
            key_point = frame_all_point[p_id]

            for i in range(18):
                center = int(float(key_point[3 * i])), int(float(key_point[3 * i + 1]))

                if center[0] == 0 and center[1] == 0:
                    continue
                cv2.circle(input_frame, center, 4, color=_color, thickness=-1)

            for limb in limbs:
                point_1 = int(float(key_point[3 * limb[0] + 0])), int(float(key_point[3 * limb[0] + 1]))
                point_2 = int(float(key_point[3 * limb[1] + 0])), int(float(key_point[3 * limb[1] + 1]))

                if point_1[0] == 0 and point_1[1] == 0:
                    continue

                if point_2[0] == 0 and point_2[0] == 0:
                    continue
                cv2.line(input_frame, point_1, point_2, _color, thickness=3)

    return input_frame

if __name__ == '__main__':

    # target = 'smoothed'
    target = 'original'

    for file_num in [39]:

        input_video_path = os.path.join(kBasePath, "avi_video", "%03d.avi" % file_num)
        point_path = os.path.join(kKeyPointPath[target], "%06d.txt" % file_num)
        result_path = os.path.join(kResultVideoBasePath, kVideoName[target] % file_num)

        keypoints = read_keypoint(point_path)

        cap = cv2.VideoCapture(input_video_path)
        width = int(cap.get(3))
        height = int(cap.get(4))
        fcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
        out = cv2.VideoWriter(result_path, fcc, 30, (width, height))
        num_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print("video %03d.avi" % file_num)
        for i in progressbar.progressbar(range(num_total_frames)):
            ret, frame = cap.read()

            if not cap.isOpened():
                print("Cannot Open Videos")

            if not ret:
                break

            frame = draw_keypoint(frame, keypoints, kPointColor[target], i)

            out.write(frame)
            cv2.waitKey(1)

        cap.release()
        out.release()


