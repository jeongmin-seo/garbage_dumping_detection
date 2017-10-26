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


def join_json_xml_file():
    for file in files:
        xml_path = "D:\etri_data\ETRIxml\ETRIxml\%03d.xml" % file

        tree = parse(xml_path)
        root = tree.getroot().find('Objects')
        for object in root.findall('Object'):
            if not int(object.find('Type').text) in [1, 111]:
                continue

            Tracks = object.find('Tracks')
            for track in Tracks.findall('Track'):
                attr = track.attrib
                json_path = "D:\etri_data\jsonfile_class1\%03d\%03d_%012d_keypoints.json" % (file, file, int(attr['frameNum']))
                people = read_pose_(json_path)['people']
                for person in people:
                    key_point = person['pose_keypoints']

                    if not (int(attr['X']) <= key_point[3] <= int(attr['X']) + int(attr['W']) and \
                                            int(attr['Y']) <= key_point[4] <= int(attr['Y']) + int(attr['H'])):
                        continue

                    data = []
                    data.append(object.find('ID').text)
                    data.append(object.find('Type').text)
                    data.append(attr['frameNum'])
                    for i in range(18):
                        data.append(str(key_point[i * 3]))
                        data.append(str(key_point[i * 3 + 1]))

                    print data
                    file_name = "%06d.txt" % file
                    if file_name in os.listdir("C:\Users\JM\Desktop\Data\ETRIrelated\pose classification\json_xml_join_file"):
                        txt_path = "C:\Users\JM\Desktop\Data\ETRIrelated\pose classification\json_xml_join_file\%s" % file_name
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
                        txt_path = "C:\Users\JM\Desktop\Data\ETRIrelated\pose classification\json_xml_join_file\%s" % file_name
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


def make_macro_file_to_dict():
    macro_path = "C:\Users\JM\Desktop\Data\ETRIrelated\pose classification\class1macro.txt"
    f = open(macro_path, 'r')

    littering_info = {}
    for lines in f.readlines():
        split_line = lines.split(" ")
        littering_info[int(split_line[0])] = [int(split_line[1]), int(split_line[2])]

    f.close()
    return littering_info


def labeling_text_file(_dict):
    text_file_path = "C:\Users\JM\Desktop\Data\ETRIrelated\pose classification\json_xml_join_file"

    for key in _dict.keys():
        file_name_check = "%06d.txt" % key

        if file_name_check not in os.listdir(text_file_path):
            continue

        text_path = "C:\Users\JM\Desktop\Data\ETRIrelated\pose classification\json_xml_join_file\%06d.txt" % key
        write_path = "C:\Users\JM\Desktop\Data\ETRIrelated\pose classification\gt_json_xml\%06d.txt" % key

        f = open(text_path, 'r')
        g = open(write_path, 'w')
        for lines in f.readlines():
            split_line = lines.split(",")
            split_line[-1] = str(float(split_line[-1]))

            if _dict[key][0] <= int(split_line[2]) <= _dict[key][1] and int(split_line[1]) == 111:
                for split in split_line:
                    g.write(split)
                    g.write(",")
                g.write("1")
                g.write("\n")

            else:
                for split in split_line:
                    g.write(split)
                    g.write(",")
                g.write("0")
                g.write("\n")
        f.close()
        g.close()

if __name__ == '__main__':
    join_json_xml_file()
    dict = make_macro_file_to_dict()
    labeling_text_file(dict)
