point = {}
import cv2

if __name__=='__main__':
    file_num = 133
    path = "C:\\Users\\JM\\Desktop\\Data\\ETRIrelated\\preprocess_data\\%06d.txt" % file_num
    f = open(path,'r')

    for line in f.readlines():
        split_line = line.split(',')
        frame_num = int(split_line[2])

        point[frame_num] = split_line[3:-1]

    video_path = "D:\\etri_tool\\101-219\\133.avi"
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(3))
    height = int(cap.get(4))
    fcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')

    out = cv2.VideoWriter('result.avi', fcc, 30, (width, height))

    num = 0
    while True:
        ret, frame = cap.read()

        if not ret:
            print("read error")
            break

        if num in point.keys():
            key_point = point[num]

            for i in range(5, 8):
                center = (int(float(key_point[2*i])), int(float(key_point[2*i+1])))
                cv2.circle(frame, center, 3, color=(0,0,255))

        num += 1
        out.write(frame)
        cv2.waitKey(1)

    cap.release()
    out.release()




