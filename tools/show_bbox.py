import cv2
import os
import argparse
import glob
import sys
import xml.etree.ElementTree as ET
from utils import print_bboxes

#ROOT_PATH = '/home/alupotto/data/autel/'


def mainParser():


    parser = argparse.ArgumentParser(description='Autel Dataset')

    parser.add_argument('--images'      , type=str,  help='path to images file')
    parser.add_argument('--video_name' , type=str, help='video to show')

    opt = parser.parse_args()

    return opt

def show_bbox(imgs_path, video_name):

    #load images
    img_files = sorted(glob.glob('%s/*.jpg' % imgs_path))

    id_img = 1
    for file_path in img_files:
        video_id = file_path.split('/')[-1].split('_')[-2]
        if video_id == video_name:
            img = cv2.imread(file_path)
            #read xml
            xml_id = file_path.replace('.jpg', '.xml')
            root = ET.parse(xml_id).getroot()

            for obj in root.findall('object'):

                obj_name = obj.find('name').text
                bndbox = obj.find('bndbox')
                x0 = float(bndbox.find('xmin').text)
                y0 = float(bndbox.find('ymin').text)
                x1 = float(bndbox.find('xmax').text)
                y1 = float(bndbox.find('ymax').text)

                bbox = cv2.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)), (0, 255, 0), 3)
                cv2.putText(bbox, obj_name, (int(x0) - 5, int(y0) - 5),
                             cv2.FONT_HERSHEY_TRIPLEX, 0.75, (0, 255, 0), 1)

            k = cv2.waitKey(250)
            if k == 27:  # If escape was pressed exit
                cv2.destroyAllWindows()
                sys.exit()
            cv2.imshow('Image', img)




if __name__ == '__main__':

    args = mainParser()
    print(args)

    show_bbox(args.images, args.video_name)
