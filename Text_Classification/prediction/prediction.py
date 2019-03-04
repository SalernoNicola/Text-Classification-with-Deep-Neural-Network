import loader.dataReaderPrediction as dataReaderPrediction

if __name__ == '__main__':
    tokenizer = 'C:\\Users\\Samsung\\Desktop\\risultatiDataScience\\tfidf7case\\tokenizer.pickle'
    dataset = 'C:\\Users\\Samsung\\Desktop\\testfile.xlsx'
    csvfile ="C:\\Users\\Samsung\\Desktop\\risultatiDataScience\\tfidf7case\\PredictionFinalissima.csv"
    modelname = "C:\\Users\\Samsung\\Desktop\\risultatiDataScience\\tfidf7case\\model.h5"
    reader = dataReaderPrediction.dataReaderPrediction(modelname,tokenizer)
    objects,labels = reader.getXlsxData(dataset) 
    #objects,labels = reader.getJsonData(dataset) 
    reader.makePrediction(objects,labels,csvfile)