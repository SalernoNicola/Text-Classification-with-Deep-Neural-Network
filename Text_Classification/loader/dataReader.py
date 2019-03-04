import pickle
import json
import random
import math
import re
from stop_words import get_stop_words
import cv2
import os
import csv

class dataReader:

    def __init__(self,): 
        self.diz_Sez = {
            "cronaca" : 0,
            "cultura" : 1,
            "economia" : 2,
            "esteri" : 3,
            "politica" : 4,
            "società" : 5,
            "sport" : 6,
            "tecnologia" : 7
            }
        self.pattern = self.getStopWords()

    def getStopWords(self):
        stop_words = get_stop_words('italian')
        pattern = re.compile(r'\b(' + r'|'.join(stop_words) + r')\b\s*')
        return pattern

    def getJsonData(self,num_labels,filename):
        with open(filename,"r", encoding="utf8") as f:
            data = json.load(f)
            train,test = self.splitDataset(data,num_labels)
            return train,test

    def splitDataset(self, data ,num_labels): 
        self.trainList = list()
        self.testList = list()   

        train=int((len(data)/num_labels) *.8)
        test=int((len(data)/num_labels)-train)
        c=train
        d=0
        for i in range(num_labels):
            c=d+train
            self.trainList=self.trainList+data[d:c]
            d=c+test
            self.testList=self.testList+data[c:d]
        
        ld = len(self.trainList)
        self.trainList = random.sample(self.trainList, ld)
        ld = len(self.testList)
        self.testList = random.sample(self.testList, ld)
        return self.trainList, self.testList

    def getValueDataset(self,listDataset):
        obsTrain = list()
        labelsTrain = list()
        for d in listDataset:
            obsTrain.append(self.pattern.sub('', d['title']) +" . "+self.pattern.sub('', d['description']))
            labelsTrain.append(self.diz_Sez[d['section']])
        return obsTrain,labelsTrain

   

