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

    parser.add_argument('--img_dir', required=True,
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
    
    img_name = []
    img_feature = []
    for file in csv_list:
        file_csv = path_dir + "/" +file
        
        #이미지 이름 저장
        strings = file.split('_feature')
        name = strings[0]
        
        read_file = cv2.FileStorage(file_csv, cv2.FILE_STORAGE_READ)
        readMat = read_file.getNode('feature').mat()
        readMat = readMat.reshape(-1,)
        
        img_name.append(name)
        img_feature.append(readMat)  
        
        task_count += 1
        
        print(f'save progress: {task_count} / {csv_num}        {int(task_count/csv_num*100)}%       image_name: {name}')
        
    print("save features complete!") 
    return img_name, img_feature
    
    
def main():
    args = parse_args()
    path_dir = args['img_dir']
    
    img_names = []
    features = []    
    img_names, features = save_features(path_dir)
    
    duplicate = []
    dup_triple_name = []
    task_count = 0
    
    for i in range(len(features) - 1):
        
        task_count += 1
        start_time = datetime.datetime.now()
        
        #socre 가 85 이상일 때 counting
        match_count = 0
        dst_list = []
        score_list = []        
        
        for j in range(i+1, len(features)):
            distance = cosine_dist(features[i], features[j])
            score = 100 - float(distance * 100)
            
            if score >= 85 :
                match_count += 1
                duplicate.append([img_names[i], img_names[j], score])
        
        if match_count >=2:
            dup_triple_name.append([img_names[i]])
            
        end_time = datetime.datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # print("progress: {task_count} / %d        %d%       장당 소요시간: %d", task_count, csv_num, task_count/csv_num, duration)
        print(f'compare progress: {task_count} / {len(features)}        {int(task_count/len(features)*100)}%       duration: {duration}           src_image_name: {img_names[i]}')
    
    with open('duplicate_images.txt', 'a') as file:
        for i in range(len(duplicate)):
            file.write(str(duplicate[i][0]) +  "    =====>    " + str(duplicate[i][1]) + "          score: " + str(duplicate[i][2]) + "\n")
        file.close()
    with open('duplicate_triple_images.txt', 'a') as file:
        for i in range(len(dup_triple_name)):
            file.write(str(dup_triple_name[i]) + "\n")
        file.close()
        
            
    print("complete!")
    
        
if __name__ == "__main__":
    main()