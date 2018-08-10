import pickle
import argparse
import glob
import sys
import xml.etree.ElementTree as ET
import cv2
import matplotlib
from tools.utils import draw_plot_func, load_classes
#matplotlib.use('agg')


def mainParser():

    parser = argparse.ArgumentParser(description='Autel Dataset')
    parser.add_argument('--images'      , type=str,  help='path to images file')
    opt = parser.parse_args()

    return opt




def check_resolutions(imgs_path):

    file_resolutions = open('files_check/resolution.txt', 'w')

    #load classes autel
    classes = load_classes('data/autel.resolutions')
    #dictionary count resolution
    dict_res_img = dict.fromkeys(classes, 0)
    dict_videos = dict.fromkeys(classes)
    dict_res_videos = {k: [] for k in dict_videos}

    #load images
    img_files = sorted(glob.glob('%s/*.jpg' % imgs_path))

    #load labels
    label_files = sorted(glob.glob('%s/*.xml' % imgs_path))

    #img with bad resolution frames
    folder_id = imgs_path.split('/')[-2]


    #img_bad_res = open("../files_check/frames_bad_{}.txt".format(folder_id), 'w')

    for img_path in img_files:

        #check if image has label
        if img_path.replace('.jpg', '.xml') in label_files:
            #video id
            video_id = img_path.split('/')[-1].split('_')[-2]

            #extract image shape
            img = cv2.imread(img_path)
            h_img, w_img, _ = img.shape

            #class id
            cls_id = resolution_id(w_img, h_img)
            class_name = classes[cls_id]

            if not video_id in dict_res_videos[class_name]:
                dict_res_videos[class_name].append(video_id)

            #num images dictionary
            if class_name in dict_res_img:
                dict_res_img[class_name] += 1

            else:
                dict_res_img[class_name] = 1

    #pickle resolution images
    with open('pickles/dict_res_img_{}.pkl'.format(folder_id), 'wb') as res_pkl:
        pickle.dump(dict_res_img, res_pkl)

    #pickle resolution videos
    with open('pickles/dict_res_videos_{}.pkl'.format(folder_id), 'wb') as res_pkl:
        pickle.dump(dict_res_videos, res_pkl)

    return dict_res_img, dict_res_videos

def histogram_resolutions_img(dict_res_img = None):


    if dict_res_img is None:
        with open('pickles/dict_res_img_part1_autel.pkl', 'rb') as gt_pkl:
            dict_res_img = pickle.load(gt_pkl)

    classes = load_classes('data/autel.resolutions')
    window_title = "img/resolution"
    plot_title = "num images per resolution"
    x_label = "num images"
    output_path = "output/resolution_img_08072018.png"
    to_show = True
    plot_color = 'royalblue'
    draw_plot_func(
        dict_res_img,
        len(classes),
        window_title,
        plot_title,
        x_label,
        output_path,
        to_show,
        plot_color,
        ""
    )

def histogram_resolutions_videos(dict_res_video = None):


    if dict_res_video is None:
        with open('pickles/dict_res_videos_part1_autel.pkl', 'rb') as gt_pkl:
            dict_res_video = pickle.load(gt_pkl)

    for key, values in dict_res_video.items():
        dict_res_video[key] = len(values)


    classes = load_classes('data/autel.resolutions')

    window_title = "ground truth"
    plot_title = "number videos per resolution"
    x_label = "num videos"
    output_path = "output/resolution_videos_08072018.png"
    to_show = True
    plot_color = 'royalblue'
    draw_plot_func(
        dict_res_video,
        len(classes),
        window_title,
        plot_title,
        x_label,
        output_path,
        to_show,
        plot_color,
        ""
    )



def resolution_id(w_img, h_img):
    cls = -1

    if w_img == 1920 and h_img == 1080:
        cls = 0

    if w_img == 1280 and h_img == 720:
        cls = 1

    if w_img == 640 and h_img == 352:
        cls = 2

    if w_img == 406 and h_img == 720:
        cls = 3

    if w_img == 960 and h_img == 720:
        cls = 4

    if w_img == 360 and h_img == 360:
        cls = 5

    if w_img == 1280 and h_img == 676:
        cls = 6

    if w_img == 360 and h_img == 204:
        cls = 7

    return cls

def main():
    args = mainParser()

    print(args)
    dict_res_img, dict_res_video = check_resolutions(args.images)


    #histograms
    histogram_resolutions_img()
    histogram_resolutions_videos()

if __name__ == '__main__':
    main()
