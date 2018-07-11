from xml.etree.ElementTree import ElementTree, parse, dump, Element, SubElement
import os
import glob
import re
import csv

xmlBasePath = "C:/Users/JM/Desktop/Data/ETRIrelated/final_xml"

# l = glob.glob(xmlBasePath, "*.xml")


def write_csv(_save_path, _save_list):
    with open(_save_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(_save_list)

for xml_name in glob.glob1(xmlBasePath, "*.xml"):
    xml_file_path = os.path.join(xmlBasePath, xml_name)

    # xml 파일 읽어오기
    tree = parse(xml_file_path)
    hole_start = int(tree.getroot().attrib['startFrameNum'])
    hole_end = int(tree.getroot().attrib['endFrameNum'])
    ground_truth = [[hole_start, hole_end]]
    csv_file_path = xml_file_path.replace('.xml', '.csv')

    verbs = tree.getroot().find('Verbs')

    if not verbs:
        write_csv(csv_file_path, ground_truth)
        continue

    for verb in verbs:
        if not int(verb.find('Type').text) == 200:
            continue

        if not verb.find('StartFrame').text:
            continue

        start = int(verb.find('StartFrame').text)
        end = int(verb.find('EndFrame').text)

        ground_truth.append([start, end])

    if not ground_truth:
        continue

    print(ground_truth)
    write_csv(csv_file_path, ground_truth)
