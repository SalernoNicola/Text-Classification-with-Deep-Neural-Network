from stop_words import get_stop_words
import json
import csv
from keras.preprocessing.text import Tokenizer
import numpy as np
import re
from sklearn.preprocessing import LabelBinarizer
from keras.models import load_model
import pickle
import openpyxl

class dataReaderPrediction:
    def __init__(self,model,savedtokenizer): 
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
        self.ar=[
            "cronaca",
            "cultura",
            "economia",
            "esteri",
            "politica",
            "società",
            "sport",
            "tecnologia"
        ]         
        self.pattern = self.getStopWords()
        #load model
        self.model = load_model(model)
        # load tokenizer
        self.tokenizer = Tokenizer()
        with open(savedtokenizer, 'rb') as handle:
            self.tokenizer = pickle.load(handle)

    def getStopWords(self):
        stop_words = get_stop_words('italian')
        pattern = re.compile(r'\b(' + r'|'.join(stop_words) + r')\b\s*')
        return pattern   

    def getXlsxData(self,filename):
        obsPrediction = list()
        labelsPrediction = list()
        book = openpyxl.load_workbook(filename)
        sheet = book.active
        cells = sheet['A2': 'C41']
        for c1, c2, c3 in cells:
            obsPrediction.append(self.pattern.sub('', c2.value) +" . "+self.pattern.sub('', c3.value))
            labelsPrediction.append(self.diz_Sez[c1.value])
        return obsPrediction,labelsPrediction

    def getJsonData(self,filename):
        with open(filename,"r", encoding="utf8") as f:
            data = json.load(f)
            return self.getValueDataset(data)

    def getValueDataset(self,listDataset):
        obsPrediction = list()
        labelsPrediction = list()
        for d in listDataset:
            obsPrediction.append(self.pattern.sub('', d['title']) +" . "+self.pattern.sub('', d['description']))
            labelsPrediction.append(self.diz_Sez[d['section']])
        return obsPrediction,labelsPrediction
 
    def makePrediction(self,obsPrediction,labelsPrediction,csvfile):
        x_test = self.tokenizer.texts_to_matrix(obsPrediction, mode='tfidf')
        csv = open(csvfile, "w") 
        columnTitleRow = "LabelOriginale, LabelPredetta\n"
        csv.write(columnTitleRow) 
        for i in range(len(x_test)):
            prediction = self.model.predict(np.array([x_test[i]])) #predizione sull'articolo
            predicted_label = self.ar[np.argmax(prediction[0])] #Restituisco l'indice con il valore massimo lungo un asse
            row = self.ar[labelsPrediction[i]] + "," + predicted_label+"\n"
            csv.write(row)       
            

    
