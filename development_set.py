import argparse
import glob
import os
import cv2
import xml.etree.ElementTree as ET
import sys
import pickle
from tools.utils import load_classes, count_classes
from random import shuffle, seed

def mainParser():


    parser = argparse.ArgumentParser(description='Autel Dataset')

    parser.add_argument('--labels_path'      , type=str,       help='path to labels ')
    parser.add_argument('--per_cent', type=float, help='path to labels ')

    opt = parser.parse_args()

    return opt


def create_dev_set(labels_path, per_cent):
    #load classes with background
    classes = load_classes('data/autel_background.names')
    #file_paths
    file_paths = open(os.path.join(os.path.dirname(labels_path.rstrip('/')), 'autel_devset.txt'), 'w')
    #load labels
    label_files = sorted(glob.glob('%s/*.txt' % labels_path))
    #check dict_classes
    if not os.path.isfile('pickles/dict_gt.pkl'):
        dict_classes = count_classes(label_files, classes)
    else:
        with open('pickles/dict_gt.pkl', 'rb') as gt_pkl:
            dict_classes = pickle.load(gt_pkl)
    #apply per_cent to classes in dict
    for key, value in dict_classes.items():
        dict_classes[key] = int(value * per_cent)

    #shuffling
    seed(23)
    shuffle(label_files)
    check_dict = {}

    for label in label_files:
        with open(label, 'r') as lbl_file:
            file = lbl_file.readlines()

        id_classes = [int(line.split(' ')[0]) for line in file]

        for cls in id_classes:
            dict_classes[classes[cls]] -= 1


        if all(value > 0 for value in dict_classes.values()):
            img_path = label.replace('labels_yolo_withbackground','images').replace('.txt','.jpg')
            file_paths.write(img_path + '\n')


def check_devset():
    # load classes with background
    classes = load_classes('data/autel_background.names')

    with open('/home/alupotto/data/autel_08102018/autel_devset.txt', 'r') as file_dev:
        list_paths = file_dev.readlines()

    lbls_path = [img_path.replace('images','labels_yolo_withbackground').replace('.jpg','.txt').strip('\n')
                    for img_path in list_paths ]

    dev_dict = count_classes(lbls_path, classes)

    print(dev_dict)


if __name__ == '__main__':

    args = mainParser()
    print(args)

    create_dev_set(args.labels_path, args.per_cent)
    #check_devset()
