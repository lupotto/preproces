import argparse
import glob
import os
import cv2
import xml.etree.ElementTree as ET
import sys
from tools.utils import load_classes, convert_yolo_sizes

def mainParser():


    parser = argparse.ArgumentParser(description='Autel Dataset')

    parser.add_argument('--labels_path'      , type=str,       help='path to labels ')

    opt = parser.parse_args()

    return opt


def file_path_images(labels_path):

    #file_paths
    file_paths = open(os.path.join(os.path.dirname(labels_path.rstrip('/')), 'autel.txt'), 'w')
    #load labels
    label_files = sorted(glob.glob('%s/*.txt' % labels_path))
    #label id
    label_id = labels_path.rstrip('/').split('/')[-1]

    for label in label_files:
        
        img_path = label.replace(label_id,'images').replace('.txt','.jpg')
        file_paths.write(img_path + '\n')

if __name__ == '__main__':

    args = mainParser()
    print(args)

    file_path_images(args.labels_path)

