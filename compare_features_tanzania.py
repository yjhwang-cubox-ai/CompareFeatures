import os
import sys 
import cv2
import csv
import pandas as pd
import datetime
from prettytable import PrettyTable
from scipy.spatial.distance import cosine

def cosine_dist(x, y):
            return cosine(x, y) * 0.5
        
def main():
    
    path_dir = 'features'
    
    csv_list = [file for file in os.listdir(path_dir) if file.endswith('.csv')]
    
    print("compare task start!")
    csv_num = len(csv_list)
    task_count = 0
    
    for file in csv_list:
        file_csv = path_dir + "/" +file
        
        task_count += 1
        start_time = datetime.datetime.now()
        
        #socre 가 85 이상일 때 counting
        match_count = 0
        
        #이미지 이름 저장
        strings = file.split('_feature')
        src = strings[0]
        
        read_file = cv2.FileStorage(file_csv, cv2.FILE_STORAGE_READ)
        readMat = read_file.getNode('feature').mat()
        readMat = readMat.reshape(-1,)
        
        dst_list = []
        score_list = []
        for file_in in csv_list:
            file_csv_in = path_dir + "/" +file_in
            #이미지 이름 저장
            strings = file_in.split('_feature')
            dst = strings[0]
            
            read_file_in = cv2.FileStorage(file_csv_in, cv2.FILE_STORAGE_READ)
            readMat_in = read_file_in.getNode('feature').mat()
            readMat_in = readMat_in.reshape(-1,)

            distance = cosine_dist(readMat, readMat_in)
            score = 100 - float(distance * 100)
            
            if score >= 85 :
                dst_list.append(dst)
                score_list.append(score)
                match_count += 1
                
        if match_count >= 2 :
            with open('duplicate_images.txt', 'a') as file:
                for i in range(len(dst_list)):
                    file.write(src + "   =====>   " + dst_list[i] + "   score: " + str(score_list[i]) + "\n")
                file.close()
                  
        read_file.release()
        read_file_in.release()
        
        end_time = datetime.datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # print("progress: {task_count} / %d        %d%       장당 소요시간: %d", task_count, csv_num, task_count/csv_num, duration)
        print(f'progress: {task_count} / {csv_num}        {int(task_count/csv_num*100)}%       duration: {duration}              image_name: {src}')
        
    print("compare task complete!\n")
        
if __name__ == "__main__":
    main()