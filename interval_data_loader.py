import json
import os
from xml.etree.ElementTree import parse

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


def read_pose_(filename):
    f = open(filename, 'r')
    js = json.loads(f.read())
    f.close()

    return js

if __name__ == '__main__':
    for file in files:
        xml_path = "D:\ETRI\ETRIxml\%03d.xml" % file

        tree = parse(xml_path)
        root = tree.getroot().find('Objects')
        for object in root.findall('Object'):
            if not int(object.find('Type').text) in [1, 111]:
                continue

            Tracks = object.find('Tracks')
            for track in Tracks.findall('Track'):
                attr = track.attrib
                json_path = "D:\ETRI\pose\%03d\%03d_%012d_keypoints.json" % (file, file, int(attr['frameNum']))
                people = read_pose_(json_path)['people']
                for person in people:
                    key_point = person['pose_keypoints']

                    """
                    check = False
                    for i in range(18):

                        if int(attr['X']) <= key_point[i*3] <= int(attr['X'])+int(attr['W']) and \
                                                int(attr['Y']) <= key_point[i*3+1] <= int(attr['Y'])+int(attr['H']):
                            continue

                        check = True
                        break
                        
                    if check:
                        continue
                    """

                    if not(int(attr['X']) <= key_point[3] <= int(attr['X']) + int(attr['W']) and \
                                            int(attr['Y']) <= key_point[4] <= int(attr['Y']) + int(attr['H'])):
                        continue

                    data = []
                    data.append(object.find('ID').text)
                    data.append(attr['frameNum'])
                    for i in range(18):
                        data.append(str(key_point[i*3]))
                        data.append(str(key_point[i*3+1]))

                    print data
                    file_name = "%06d.txt" % file
                    if file_name in os.listdir("D:\workspace\data\etri"):
                        txt_path = "D:\workspace\data\etri\%s" % file_name
                        f = open(txt_path, 'a')

                        iter = 1
                        for dat in data:
                            f.write(dat)
                            if len(data) == iter:
                                f.write("\n")
                                continue
                            f.write(",")
                            iter += 1
                        f.close()

                    else:
                        txt_path = "D:\workspace\data\etri\%s" % file_name
                        f = open(txt_path, 'w')

                        iter = 1
                        for dat in data:
                            f.write(dat)
                            if len(data) == iter:
                                f.write("\n")
                                continue
                            f.write(",")
                            iter += 1
                        f.close()

    """
    xml_path = "D:\ETRI\ETRIxml\%03d.xml" % 6

    tree = parse(xml_path)
    root = tree.getroot().find('Objects')
    for object in root.findall('Object'):
        if not int(object.find('Type').text) in [1, 111]:
            continue

        Tracks = object.find('Tracks')
        for track in Tracks.findall('Track'):
            attr = track.attrib
            json_path = "D:\ETRI\pose\%03d\%03d_%012d_keypoints.json" % (6, 6, int(attr['frameNum']))
            people = read_pose_(json_path)['people']
            for person in people:
                key_point = person['pose_keypoints']

                check = False
                for i in range(18):

                    if int(attr['X']) <= key_point[i*3] <= int(attr['X'])+int(attr['W']) and \
                                            int(attr['Y']) <= key_point[i*3+1] <= int(attr['Y'])+int(attr['H']):
                        print("continue")
                        continue

                    check = True
                    break

                if check:
                    print(1)
                    continue

                data = []
                data.append(object.find('ID').text)
                data.append(attr['frameNum'])
                for i in range(18):
                    data.append(str(key_point[i*3]))
                    data.append(str(key_point[i*3+1]))

                print(data)

                file_name = "%06d.txt" % 6
                if file_name in os.listdir("D:\workspace\data\etri"):
                    txt_path = "D:\workspace\data\etri\%s" % file_name
                    f = open(txt_path, 'a')

                    for dat in data:
                        f.write(dat)
                        if data[-1] == dat:
                            f.write("\n")
                            continue
                        f.write(",")
                    f.close()

                else:
                    txt_path = "D:\workspace\data\etri\%s" % file_name
                    f = open(txt_path, 'w')

                    for dat in data:
                        f.write(dat)
                        if data[-1] == dat:
                            f.write("\n")
                            continue
                        f.write(",")
                    f.close()
    """
