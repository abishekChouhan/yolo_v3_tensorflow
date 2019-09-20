# coding: utf-8

import xml.etree.ElementTree as ET
import os

names_dict = {'Apple': 0, 'Banana':1, 'Orange':2, 'Capsicum':3, 'Tomato':4, 'Potato':5}

def parse_xml(path):
    tree = ET.parse(path)
    img_name = path.split('/')[-1][:-4]
    
    height = tree.findtext("./size/height")
    width = tree.findtext("./size/width")

    objects = [img_name, width, height]

    for obj in tree.findall('object'):
        difficult = obj.find('difficult').text
        if difficult == '1':
            continue
        name = str(obj.find('name').text).capitalize()
        bbox = obj.find('bndbox')
        xmin = bbox.find('xmin').text
        ymin = bbox.find('ymin').text
        xmax = bbox.find('xmax').text
        ymax = bbox.find('ymax').text

        name = str(names_dict[name])
        objects.extend([name, xmin, ymin, xmax, ymax])
    if len(objects) > 1:
        return objects
    else:
        return None


def gen_txt(txt_path):
    cnt = 0
    f = open(txt_path+'.txt', 'w')

    files = os.listdir(txt_path+'/')
    for file in files:
        if file[-3:] == 'xml':
            xml_path = txt_path + '/' + file
            objects = parse_xml(xml_path)
            if objects:
                objects[0] = txt_path + '/' + file.split('.')[0] + '.jpg'
                if os.path.exists(objects[0]):
                    objects.insert(0, str(cnt))
                    cnt += 1
                    objects = ' '.join(objects) + '\n'
                    f.write(objects)
    f.close() 



gen_txt('train')
gen_txt('test')


