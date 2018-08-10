import argparse
import os
import sys
from utils import create_labels, stats_gt, train_test, check_labels,create_file_paths


def mainParser():
    parser = argparse.ArgumentParser(description='Autel Dataset')
    parser.add_argument('--file_path', type=str, default='data/autel/images', help='path dataset folder')
    parser.add_argument('--img_path', type=str, default='data/autel/images', help='path dataset folder')
    parser.add_argument('--labels_path', type=str, default=False, help='path to label creation')
    parser.add_argument('--class_path', type=str, default='data/autel.names', help='path to label creation')
    parser.add_argument('--create_stats', type=bool, default=False, help='path to label creation')

    return parser

def main():

    args = mainParser()
    print(args)



    #config = importlib.import_module(params_path[:-3]).PREPROCESS_PARAMS


    create_labels(config["images"], config["label_path"], config["class_path"])

    if config["create_file_paths"]:
        create_file_paths(config["images"])

    if config["create_histograms"]:
        stats_gt(config["label_path"], config["class_path"])
        #TODO: histogram
        #classwise_histogram(config["class_path"])

    if config["split_data"]:
        train_test(config["total_file"], config["label_path"])

    if config["check_labels"]:
        check_labels(config["total_file"], config["label_path"])




if __name__ == '__main__':

    main()




