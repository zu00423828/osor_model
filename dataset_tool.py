import csv
import os
from random import randint
import cv2
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset
# import dlib

augment=transforms.Compose([transforms.ToTensor()])
class TestDataSet(Dataset):
    def __init__(self, root,split):
        self.filelist,self.video_name= self.get_file_list(root,split)
    def __len__(self):
        return len(self.video_name)
    def __getitem__(self, index):
        rand_idx=self.random_get(index,self.video_name)
        # landmark_img=cv2.imread(self.landmark_list[index],cv2.IMREAD_GRAYSCALE)
        reference_img=cv2.imread(self.filelist[rand_idx][1])
        target_img=cv2.imread(self.filelist[index][1])
     
        reference_img=augment(reference_img)
        target_img=augment(target_img)
        reference_img=(reference_img-0.5)/0.5
        target_img=(target_img-0.5)/0.5
        return reference_img,target_img

    def get_file_list(self,root,split):
        filelist=[]
        videoname=[]
        with open(f"{root}/{split}_list.csv",newline='')as csvfile:
            rows = csv.reader(csvfile)
            path_join=os.path.join
            filelist_append=filelist.append
            videoname_append=videoname.append
            for row in rows:
                filelist_append([path_join(root,file)for file in row])
                videoname_append(row[0].split("/")[-2])
        return filelist,videoname
    def random_get(self,idx,video_name):
        min_idx=video_name.index(video_name[idx])
        max_idx=len(video_name)-1-video_name[::-1].index(video_name[idx])
        r_idx=randint(min_idx,max_idx)
        r_idx=randint(min_idx,max_idx)
        while idx==r_idx:
            r_idx=randint(min_idx,max_idx)
        return  r_idx
    def random_feture(self,idx,video_name):
        min_idx=video_name.index(video_name[idx])
        max_idx=len(video_name)-1-video_name[::-1].index(video_name[idx])
        r_idx=randint(min_idx,max_idx)
        featurelist=[]
        while len(featurelist)<8:
            if idx!=r_idx and r_idx not in featurelist:
                    featurelist.append(r_idx)
            r_idx=randint(min_idx,max_idx)
        feature_img=None
        for i in featurelist:
            img=cv2.imread(self.filelist[i][1])
            if feature_img is None:
                feature_img=img
            else:
                feature_img=np.concatenate((feature_img,img),axis=2)
        feature_img=cv2.resize(feature_img,self.resize)
        return  feature_img

    
if __name__=='__main__':
    # transform=transforms.Compose([transforms.ToTensor()])
    # dataset=BgMixerDataset("raw",256,transform)
    # print(len(dataset))
    # a = TestDataSet("/home/yuan/hdd/avspeech_preprocess/test/train")
    # print(len(a))
    # a = TestDataSet("/home/yuan/hdd/avspeech_preprocess/train")
    train_dataset=TestDataSet("/home/yuan/hdd/avspeech_preprocess/preprocess4","train",resize=256)
    print(train_dataset[0][-1].shape)
 