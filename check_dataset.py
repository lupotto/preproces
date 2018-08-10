import pickle
import argparse
import glob
import sys
import xml.etree.ElementTree as ET
import cv2
import matplotlib
from tools.utils import draw_plot_func, load_classes
matplotlib.use('agg')


def mainParser():

    parser = argparse.ArgumentParser(description='Autel Dataset')
    parser.add_argument('--images'      , type=str,  help='path to images file')
    opt = parser.parse_args()

    return opt

def check_labels(imgs_path):

    #load classes autel
    classes = load_classes('data/autel.names')

    #dictionary count classes
    dict_gt = dict.fromkeys(classes, 0)

    #dictionary count resolution
    dict_resolution = {}

    #load images
    img_files = sorted(glob.glob('%s/*.jpg' % imgs_path))

    #load labels
    label_files = sorted(glob.glob('%s/*.xml' % imgs_path))

    #img with bad resolution frames
    folder_id = imgs_path.split('/')[-2]
    img_bad_res = open("files_check/frames_bad_{}.txt".format(folder_id), 'w')

    for img_path in img_files:
       # print(dict_gt)
        #check if image has label
        if img_path.replace('.jpg', '.xml') in label_files:
            #xml path
            xml_path = img_path.replace('.jpg', '.xml')

            #root xml
            root = ET.parse(xml_path)

            #check real image shape with shape of the label
            h_xml = int(root.find('size').find('height').text)
            w_xml = int(root.find('size').find('width').text)

            #extract image shape
            img = cv2.imread(img_path)
            h_img, w_img, ch_img = img.shape


            #check xml & jpg are equal

            if check_size_objects((h_img, w_img), (h_xml, w_xml),  root):

                for obj in root.findall('object'):
                    obj_name = obj.find('name').text
                    #count objects
                    if obj_name in dict_gt:
                        dict_gt[obj_name] += 1
                    else:
                        dict_gt[obj_name] = 1


    with open('pickles/dict_{}.pkl'.format(folder_id), 'wb') as gt_pkl:
        pickle.dump(dict_gt, gt_pkl)



def histogram_classes_gt(imgs_path, dict_gt = None):


    if dict_gt is None:
        with open('pickles/dict_part1_autel.pkl', 'rb') as gt_pkl:
            dict_gt = pickle.load(gt_pkl)

    print(dict_gt)

    classes = load_classes('data/autel.names')
    # load images
    img_files = sorted(glob.glob('%s/*.jpg' % imgs_path))
    print(len(img_files))
    window_title = "ground truth"
    plot_title = "Ground-Truth\n"
    plot_title += "(" + str(len(img_files)) + " files and " + str(len(classes)) + " classes)"
    x_label = "number ground truth objects"
    output_path = "output/ground_truth_08072018.png"
    to_show = True
    plot_color = 'forestgreen'
    draw_plot_func(
        dict_gt,
        len(classes),
        window_title,
        plot_title,
        x_label,
        output_path,
        to_show,
        plot_color,
        ""
    )

def check_size_objects(img_size, xml_size, root):

    check = False

    #w_img = 1280
    #h_img = 720

    h_img, w_img = img_size
    h_xml, w_xml = xml_size

    if h_xml == h_img and w_xml == w_img and len(root.findall('object')) > 0:
        check = True

    return check



def main():
    args = mainParser()

    print(args)
    dict_gt = check_labels(args.images)
    histogram_classes_gt(args.images)

if __name__ == '__main__':
    main()
