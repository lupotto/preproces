import matplotlib.pyplot as plt
import operator
import cv2
import sys
import os
import pickle
from sklearn.model_selection import train_test_split

def load_classes(classes_path):

    fp = open(classes_path, "r")
    names = fp.read().split("\n")[:-1]

    return names

def draw_plot_func(dictionary, n_classes, window_title, plot_title, x_label, output_path, to_show, plot_color,
                   true_p_bar):
    # sort the dictionary by decreasing value, into a list of tuples
    sorted_dic_by_value = sorted(dictionary.items(), key=operator.itemgetter(1))
    # unpacking the list of tuples into two lists
    sorted_keys, sorted_values = zip(*sorted_dic_by_value)
    #
    if true_p_bar != "":
        """
         Special case to draw in (green=true predictions) & (red=false predictions)
        """
        fp_sorted = []
        tp_sorted = []
        for key in sorted_keys:
            fp_sorted.append(dictionary[key] - true_p_bar[key])
            tp_sorted.append(true_p_bar[key])
        plt.barh(range(n_classes), fp_sorted, align='center', color='crimson', label='False Predictions')
        plt.barh(range(n_classes), tp_sorted, align='center', color='forestgreen', label='True Predictions',
                 left=fp_sorted)
        # add legend
        plt.legend(loc='lower right')
        """
         Write number on side of bar
        """
        fig = plt.gcf()  # gcf - get current figure
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
            if i == (len(sorted_values) - 1):  # largest bar
                adjust_axes(r, t, fig, axes)
    else:
        plt.barh(range(n_classes), sorted_values, color=plot_color)
        """
         Write number on side of bar
        """
        fig = plt.gcf()  # gcf - get current figure
        axes = plt.gca()
        r = fig.canvas.get_renderer()
        for i, val in enumerate(sorted_values):
            str_val = " " + str(val)  # add a space before
            if val < 1.0:
                str_val = " {0:.2f}".format(val)
            t = plt.text(val, i, str_val, color=plot_color, va='center', fontweight='bold')
            # re-set axes to show number inside the figure
            if i == (len(sorted_values) - 1):  # largest bar
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
    height_pt = n_classes * (tick_font_size * 1.4)  # 1.4 (some spacing)
    height_in = height_pt / dpi
    # compute the required figure height
    top_margin = 0.15  # in percentage of the figure height
    bottom_margin = 0.05  # in percentage of the figure height
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

def count_classes(list_labels, classes):

    dict_classes = {}

    for label in list_labels:
        #open label file
        with open(label, 'r') as lbl_file:
            lines = lbl_file.readlines()

        id_classes = [int(line.split(' ')[0]) for line in lines]

        for cls in id_classes:
            if classes[cls] in dict_classes:
                dict_classes[classes[cls]] += 1
            else:
                dict_classes[classes[cls]] = 1

    with open('pickles/dict_gt.pkl', 'wb') as gt_pkl:
        pickle.dump(dict_classes, gt_pkl)

    return dict_classes