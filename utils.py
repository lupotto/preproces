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


def draw_plot_func(dictionary, n_classes, true_p_bar, gt_labels):

    results_files_path = 'auteltools'
    window_title = "autel_ground_truth"
    plot_title = "Ground-Truth\n"
    plot_title += "(" + str(len(gt_labels)) + " files and " + str(n_classes) + " classes)"
    x_label = "Number of objects per class"
    output_path = "histograms/autel_ground_truth_new.png"
    to_show = False
    plot_color = 'forestgreen'

    # sort the dictionary by decreasing value, into a list of tuples
    sorted_dic_by_value = sorted(dictionary.items(), key=operator.itemgetter(1))
    # unpacking the list of tuples into two lists
    sorted_keys, sorted_values = zip(*sorted_dic_by_value)
    #
    if true_p_bar != 0:
        """
         Special case to draw in (green=true predictions) & (red=false predictions)
        """
        fp_sorted = []
        tp_sorted = []
        for key in sorted_keys:
          fp_sorted.append(dictionary[key] - true_p_bar[key])
          tp_sorted.append(true_p_bar[key])
        plt.barh(range(n_classes), fp_sorted, align='center', color='crimson', label='False Predictions')
        plt.barh(range(n_classes), tp_sorted, align='center', color='forestgreen', label='True Predictions', left=fp_sorted)
        # add legend
        plt.legend(loc='lower right')
        """
         Write number on side of bar
        """
        fig = plt.gcf() # gcf - get current figure
        axes = plt.gca()
        r = fig.canvas.get_renderer()
        for i, val in enumerate(sorted_values):
          fp_val = fp_sorted[i]
          tp_val = tp_sorted[i]
          fp_str_val = " " + str(fp_val)
          tp_str_val = fp_str_val + " " + str(tp_val)
          # trick to paint multicolor with offset:
          #   first paint everything and then repaint the first number
          t = plt.text(val, i, tp_str_val, color='forestgreen', va='center', fontweight='bold')
          plt.text(val, i, fp_str_val, color='crimson', va='center', fontweight='bold')
          if i == (len(sorted_values)-1): # largest bar
            adjust_axes(r, t, fig, axes)
    else:
        plt.barh(range(n_classes), sorted_values, color=plot_color)
        """
         Write number on side of bar
        """
        fig = plt.gcf() # gcf - get current figure
        axes = plt.gca()
        r = fig.canvas.get_renderer()
        for i, val in enumerate(sorted_values):
          str_val = " " + str(val) # add a space before
          if val < 1.0:
            str_val = " {0:.2f}".format(val)
          t = plt.text(val, i, str_val, color=plot_color, va='center', fontweight='bold')
          # re-set axes to show number inside the figure
          if i == (len(sorted_values)-1): # largest bar
            adjust_axes(r, t, fig, axes)

    # set window title
    fig.canvas.set_window_title(window_title)
    # write classes in y axis
    tick_font_size = 12
    plt.yticks(range(n_classes), sorted_keys, fontsize=tick_font_size)
    """
    Re-scale height accordingly
    """
    init_height = fig.get_figheight()
    # comput the matrix height in points and inches
    dpi = fig.dpi
    height_pt = n_classes * (tick_font_size * 1.4) # 1.4 (some spacing)
    height_in = height_pt / dpi
    # compute the required figure height
    top_margin = 0.15    # in percentage of the figure height
    bottom_margin = 0.05 # in percentage of the figure height
    figure_height = height_in / (1 - top_margin - bottom_margin)
    # set new height
    if figure_height > init_height:
        fig.set_figheight(figure_height)

    # set plot title
    plt.title(plot_title, fontsize=14)
    # set axis titles
    # plt.xlabel('classes')
    plt.xlabel(x_label, fontsize='large')
    # adjust size of window
    fig.tight_layout()

    # save the plot
    fig.savefig(output_path)
    # show image
    if to_show:
        plt.show()
    # close the plot
    plt.close()


def adjust_axes(r, t, fig, axes):
    # get text width for re-scaling
    bb = t.get_window_extent(renderer=r)
    text_width_inches = bb.width / fig.dpi
    # get axis width in inches
    current_fig_width = fig.get_figwidth()
    new_fig_width = current_fig_width + text_width_inches
    propotion = new_fig_width / current_fig_width
    # get axis limit
    x_lim = axes.get_xlim()
    axes.set_xlim([x_lim[0], x_lim[1]*propotion])

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



