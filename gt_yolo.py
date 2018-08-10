import argparse
import glob
import os
import cv2
import xml.etree.ElementTree as ET
import sys
from tools.utils import load_classes, convert_yolo_sizes

def mainParser():


    parser = argparse.ArgumentParser(description='Autel Dataset')

    parser.add_argument('--images_path'      , type=str,  help='path to images folder')
    parser.add_argument('--labels_path'      , type=str,       help='path to labels ')

    opt = parser.parse_args()

    return opt


def create_labels_yolo(images_path, labels_path):

    #load classes autel
    classes = load_classes('data/autel.names')

    #dictionary count classes
    dict_gt = dict.fromkeys(classes, 0)

    #load images
    img_files = sorted(glob.glob('%s/*.jpg' % images_path))

    #load labels
    label_files = sorted(glob.glob('%s/*.xml' % images_path))

    for img_path in img_files:
        video_id = img_path.split('/')[-1].split('_')[-2]

        if img_path.replace('.jpg', '.xml') in label_files:

            # xml path
            xml_path = img_path.replace('.jpg', '.xml')

            # root xml
            root = ET.parse(xml_path)

            # check real image shape with shape of the label
            h_xml = int(root.find('size').find('height').text)
            w_xml = int(root.find('size').find('width').text)

            # extract image shape
            img = cv2.imread(img_path)
            h_img, w_img, _ = img.shape

            #check resolution objects
            if check_resolution(h_img,w_img, video_id):

                #label file generation
                label_id = img_path.replace('.jpg','.txt').split('/')[-1]
                #prompt
                print(labels_path + label_id)
                gt_file = open(labels_path + label_id, 'w')

                for obj in root.findall('object'):
                    obj_name = obj.find('name').text

                    if obj_name in classes:
                        #parse xml
                        bndbox = obj.find('bndbox')
                        x0 = float(bndbox.find('xmin').text)
                        y0 = float(bndbox.find('ymin').text)
                        x1 = float(bndbox.find('xmax').text)
                        y1 = float(bndbox.find('ymax').text)
                        #prepare coords for conversion
                        obj_id = classes.index(obj_name)
                        bbox_xml = (x0, x1, y0, y1)
                        #convert to yolo sizes
                        bbox_yolo = convert_yolo_sizes((w_img, h_img), bbox_xml)
                        #print in txt file
                        gt_file.write(
                            str(obj_id) + " " + " ".join([str(coord) for coord in bbox_yolo]) + '\n')

def check_resolution(h_img, w_img, video_id):

    #Good resolution: 1280x720 / 1920x1080 / 960x720 (Vid0030) / 1280 x 676
    #Bad resolution:  640x352 / 406x720 / 960x720 (Vid0035) / 360x360 / 204x360

    if w_img == 1280 and h_img == 720 or w_img == 1920 and h_img == 1080  \
                or w_img == 960 and h_img == 720 and video_id == 'Vid0030'\
                or w_img == 1280 and h_img == 676:
        return True

    return False


if __name__ == '__main__':

    args = mainParser()
    print(args)

    create_labels_yolo(args.images_path, args.labels_path)

