import os
import glob
import sys
import xml.etree.ElementTree as ET
import numpy as np
import cv2
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import operator
import warnings
from sklearn.model_selection import train_test_split
from PIL import Image


def create_labels(imgs_path, labels_path, classes_path):

    classes = load_classes(classes_path)
    imgs_autel = open(os.path.dirname(imgs_path.rstrip('/'))+"/total_autel_local.txt",'w')
   # wrong_labels = open(os.path.dirname(imgs_path.rstrip('/'))+"/wrong_labels_object.txt",'w')

    img_files = sorted(glob.glob('%s**/*.jpg' % imgs_path))
    label_files = sorted(glob.glob('%s**/*.xml' % imgs_path))
    i = 0
    x = 0
    for img_path in img_files:
        # TODO: optimize with os.path.exists or with dictionary and parse_labels in more functions
        if img_path.replace('.jpg', '.xml') in label_files:
            check = parse_labels(img_path, img_path.replace('.jpg', '.xml'), classes, labels_path)
            if check:
                imgs_autel.write(img_path + '\n')
                print("{}: {}".format(i,img_path))
                i += 1
            else:

     #           wrg_path = os.path.join(os.path.basename(os.path.dirname(img_path)),
      #                                  os.path.basename(img_path.replace('.jpg','.xml')))
    #            wrong_labels.write('{}: {}\n'.format(x, wrg_path))

                x += 1
        else:
            continue
           # wrg_path = os.path.join(os.path.basename(os.path.dirname(img_path)),
            #                        os.path.basename(img_path.replace('.jpg','.xml')))

           # wrong_labels.write('{}: {}\n'.format(i, wrg_path))

def create_file_paths(imgs_path):
    img_files = sorted(glob.glob('%s**/*.jpg' % imgs_path))

    out_file_paths = open(os.path.dirname(imgs_path.rstrip('/')) + "/train_autel_local.txt", 'w')
    for imgs_path in img_files:
        print(imgs_path)
        out_file_paths.write("{}\n".format(imgs_path))




def parse_labels(img_path, xml, classes, labels_path):
    check = False
    root = ET.parse(xml).getroot()
    #check real image shape with shape of the label
    h_xml = int(root.find('size').find('height').text)
    w_xml = int(root.find('size').find('width').text)

    img = np.array(Image.open(img_path))
    h_img, w_img, _ = img.shape

    check = check_size_objects((h_img, w_img), (h_xml, w_xml),  root)

    if check:
        #out_file = open(labels_path+img_path.replace('.jpg', '.txt').split('/')[-1], 'w')

        for obj in root.findall('object'):
            obj_name = obj.find('name').text

            if obj_name in classes:
                bndbox = obj.find('bndbox')
                x0 = float(bndbox.find('xmin').text)
                y0 = float(bndbox.find('ymin').text)
                x1 = float(bndbox.find('xmax').text)
                y1 = float(bndbox.find('ymax').text)

                idx_class = classes.index(obj_name)
                bbox_xml = (x0, x1, y0, y1)
                bbox_yolo = convert_yolo_sizes((w_img, h_img), bbox_xml)

                #TODO: scalable
                #print_bboxes(bbox_xml,obj_name,img_path)

                #out_file.write(str(idx_class) + " " + " ".join([str(coord) for coord in bbox_yolo]) + '\n')
    return check

def check_size_objects(img_size, xml_size, root):
    check = False

    h_img, w_img = img_size
    h_xml, w_xml = xml_size

    if h_xml == h_img and w_xml == w_img and len(root.findall('object')) > 0:
        check = True

    return check

def check_labels(imgs_file, labels_path):
    with open(imgs_file, 'r') as file:
        img_files = file.readlines()

    label_files = sorted(glob.glob('%s/*.txt' % labels_path))
    i = 0
    x = 0
    img =img_files[0].split('/')[-1].strip('\n')
    lbl_files = [path_label.split('/')[-1].strip('\n') for path_label in label_files]
    dict_paths = dict()

    for path in img_files:
        big_path = path
        path = path.split('/')[-1].strip('\n')
        if path in dict_paths:
            print(big_path.split('/')[-2])
            dict_paths[path] += 1
        else:
            dict_paths[path] = 0

    d = dict((k, v) for k, v in dict_paths.items() if v > 1)
    print(len(d))
    print(sum(d.values()))
    print("labels equal {}".format(x))
    print(dict_paths)
    print(len(img_files) - len(label_files))
    # load labels
    print(len(img_files))
    print(len(label_files))

def convert_yolo_sizes(image_size, bbox_xml):

    dw = 1. / (image_size[0])
    dh = 1. / (image_size[1])
    x = (bbox_xml[0] + bbox_xml[1]) / 2.0 - 1
    y = (bbox_xml[2] + bbox_xml[3]) / 2.0 - 1
    w = bbox_xml[1] - bbox_xml[0]
    h = bbox_xml[3] - bbox_xml[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh

    return (x, y, w, h)

def load_classes(classes_path):

    fp = open(classes_path, "r")
    names = fp.read().split("\n")[:-1]

    return names

def print_bboxes(b,name,path_file):

    font_color = (0, 0, 0)
    font = cv2.FONT_HERSHEY_TRIPLEX
    font_scale = 0.75
    line_type = 1
    #b[0] xmin b[1] xmax
    #b[2] ymin b[3] ymax
    img = cv2.imread(path_file)
    #font_color = set_color(name)
    bbox = cv2.rectangle(img,(int(b[0]),int(b[2])),(int(b[1]),int(b[3])),font_color,2)
    newimg = cv2.putText(bbox, name, (int(b[0]) - 5, int(b[2]) - 5),font,
                                    font_scale, font_color, line_type)
    k = cv2.waitKey(1)

    if k == 27:  # If escape was pressed exit
        cv2.destroyAllWindows()
        sys.exit()

    cv2.imshow('Image', newimg)

def stats_gt(labels_path, class_path):


    gt_labels = sorted(glob.glob('%s*.txt' % labels_path))

    classes = load_classes(class_path)
    dict_classes = dict.fromkeys(classes,0)

    for file in gt_labels:

        warnings.simplefilter("ignore")
        label = np.loadtxt(os.path.join(labels_path, file),ndmin=2)

        if len(label) is not 0:
            idx_cls = label[:, 0].astype(int)

            #name_cls = [dict_classes[classes[x]] for x in idx_cls]
            name_cls = [classes[x] for x in idx_cls]
            for cls in name_cls:
                dict_classes[cls] += 1

    draw_plot_func(dict_classes, len(classes), 0, gt_labels)


def get_dict_gt(labels_path, class_path):


    gt_labels = sorted(glob.glob('%s*.txt' % labels_path))
    classes = load_classes(class_path)
    dict_classes = dict.fromkeys(classes,0)

    for file in gt_labels:
        #TODO: empty input files
        label = np.loadtxt(os.path.join(labels_path, file),ndmin=2)
        if len(label) is not 0:
            idx_cls = label[:, 0].astype(int)
            #TODO: improve loop optimized
            #name_cls = [dict_classes[classes[x]] for x in idx_cls]
            name_cls = [classes[x] for x in idx_cls]
            for cls in name_cls:
                dict_classes[cls] += 1

    return dict_classes, list(dict_classes.values())


def train_test(total_file , label_path):


    train_autel = open(os.path.dirname(total_file.rstrip('/')) + "/train_autel.txt", 'w')
    test_autel = open(os.path.dirname(total_file.rstrip('/')) + "/test_autel.txt", 'w')

    with open(total_file, 'r') as file:
          img_files = file.readlines()

    label_files = [os.path.join(label_path, path.split('/')[-1].replace('.jpg', '.txt').rstrip('\n'))
                        for path in img_files]


    X_train, X_test, _, _ = train_test_split(img_files, label_files, test_size=0.1, random_state=42)

    for path in X_train:
        train_autel.write(path)

    for path in X_test:
        test_autel.write(path)


def convert_yolo_coordinates_to_voc(x_c_n, y_c_n, width_n, height_n, img_width, img_height):
    ## remove normalization given the size of the image
    x_c = float(x_c_n) * img_width
    y_c = float(y_c_n) * img_height
    width = float(width_n) * img_width
    height = float(height_n) * img_height

    ## compute half width and half height
    half_width = width / 2
    half_height = height / 2

    ## compute left, top, right, bottom
    ## in the official VOC challenge the top-left pixel in the image has coordinates (1;1)
    left = int(x_c - half_width) + 1
    top = int(y_c - half_height) + 1
    right = int(x_c + half_width) + 1
    bottom = int(y_c + half_height) + 1

    return left, top, right, bottom

def convert_yolo_to_voc_gt(labels_origin, label_dest, file_paths, class_path):

    #read images
    with open(file_paths) as file:
        img_files = file.readlines()
    #read labels
    label_files = [os.path.join(labels_origin, path.split('/')[-1].replace('.jpg', '.txt').rstrip('\n'))
                        for path in img_files]
    #read classes
    classes_list = load_classes(class_path)

    for num, img_path in enumerate(img_files):


        img = cv2.imread(img_path.strip('\n'))
        img_h, img_w = img.shape[:2]



        for label in label_files:
            label_name = label.split('/')[-1]
           # new_file = open('{}{}'.format(label_dest,label_name),'w')

            with open(label) as f_label:
                content = f_label.readlines()

            # remove whitespace characters like `\n` at the end of each line
            content = [x.strip() for x in content]

            for line in content:
                ## split a line by spaces.
                ## "c" stands for center and "n" stands for normalized
                obj_id, x_c_n, y_c_n, width_n, height_n = line.split()
                obj_name = classes_list[int(obj_id)]

                left, top, right, bottom = convert_yolo_coordinates_to_voc(x_c_n, y_c_n, width_n, height_n, img_h,
                                                                           img_w)
                ## add new line to file
                # print(obj_name + " " + str(left) + " " + str(top) + " " + str(right) + " " + str(bottom))
                print(left,top,right,bottom)
                print_bboxes((left,top,right,bottom), obj_name, img_path.strip('\n'))
                #new_file.write(obj_name + " " + str(left) + " " + str(top) + " " + str(right) + " " + str(bottom) + '\n')

