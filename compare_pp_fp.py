import os
import sys 
import cv2
import csv
import argparse
import pandas as pd
import datetime
from prettytable import PrettyTable
from scipy.spatial.distance import cosine

def parse_args():

    parser = argparse.ArgumentParser(description='compare features')

    parser.add_argument('--gallery_dir', required=True,
                        help='Name of the images directory', default="features")
    parser.add_argument('--feature_dir', required=True,
                        help='Name of the images directory', default="test")

    args = vars(parser.parse_args())
    return args

def cosine_dist(x, y):
            return cosine(x, y) * 0.5
        
def save_features(img_dir):
    
    path_dir = img_dir
    
    csv_list = [file for file in os.listdir(path_dir) if file.endswith('.csv')]
    
    print("save features start!")
    csv_num = len(csv_list)
    task_count = 0
    
    dict_feature = {}    
    for file in csv_list:
        file_csv = path_dir + "/" +file
        
        #이미지 이름 저장
        strings = file.split('_feature')
        src = strings[0]
        
        read_file = cv2.FileStorage(file_csv, cv2.FILE_STORAGE_READ)
        readMat = read_file.getNode('feature').mat()
        readMat = readMat.reshape(-1,)
        
        dict_feature[src] = readMat        
        
        task_count += 1
        
        print(f'save progress: {task_count} / {csv_num}        {int(task_count/csv_num*100)}%       image_name: {src}')
        
    print("save features complete!") 
    return dict_feature
    
    
def main():
    args = parse_args()
    gallery_dir = args['gallery_dir']
    feature_dir = args['feature_dir']
    
        
    gallery_features = {}
    features = {}
    gallery_features = save_features(gallery_dir)
    features = save_features(feature_dir)    
    
    duplicate = []
    task_count = 0
    
    for src_img_name, src_feature in features.items():
        
        task_count += 1
        start_time = datetime.datetime.now()
        
        #socre 가 85 이상일 때 counting
        match_count = 0
        dst_list = []
        score_list = []
        
        for dst_img_name, dst_feature in gallery_features.items():
            distance = cosine_dist(src_feature, dst_feature)
            score = 100 - float(distance * 100)
            
            if score >= 85 :
                match_count += 1
                duplicate.append([src_img_name, dst_img_name, score])
            
        end_time = datetime.datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # print("progress: {task_count} / %d        %d%       장당 소요시간: %d", task_count, csv_num, task_count/csv_num, duration)
        print(f'compare progress: {task_count} / {len(features.items())}        {int(task_count/len(features.items())*100)}%       duration: {duration}           src_image_name: {src_img_name}')
        print(f'match count: {match_count}')
    
    with open('fp_duplicate_images.txt', 'a') as file:
        for item in duplicate:
            file.write(str(item[0]) +  "    =====>    " + str(item[1][0]) + ",  " + str(item[1][1]) + "\n")
        file.close()
            
    print("complete!")
    
        
if __name__ == "__main__":
    main()