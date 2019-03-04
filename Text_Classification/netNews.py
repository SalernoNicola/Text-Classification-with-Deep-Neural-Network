import numpy as np
import pickle
from keras.optimizers import Adam
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout
from sklearn.preprocessing import LabelBinarizer
from keras.utils import np_utils
import os
from loader.dataReader import dataReader

#Tokenizzo e preparo il vocabolario
# define Tokenizer with Vocab Size, Tokenizer viene utilizzato per convertire il testo in vettore numerico
#texts_to_matrix fa lo stesso di tokenizer
def makeTokenize(obsTrain,obsTest,labelsTrain,labelsTest,num_labels,vocab_size,mode):
    tokenizer = Tokenizer(num_words=vocab_size) # prende le prime 150000 parole
    tokenizer.fit_on_texts(obsTrain) 

    x_train = tokenizer.texts_to_matrix(obsTrain, mode=mode)
    x_test = tokenizer.texts_to_matrix(obsTest, mode=mode)

    y_train = np_utils.to_categorical(labelsTrain, num_labels)
    y_test = np_utils.to_categorical(labelsTest, num_labels)
    return x_train,x_test,y_train,y_test,tokenizer

##### Costruzione della rete #####
def build_model(num_labels,vocab_size):
    model = Sequential()
    model.add(Dense(2048, input_shape=(vocab_size,)))
    model.add(Activation('relu'))

    model.add(Dense(1024))
    model.add(Activation('relu'))
    
    model.add(Dropout(0.3))
    
    model.add(Dense(512))
    model.add(Activation('relu'))
    
    model.add(Dropout(0.3))
    
    model.add(Dense(512))
    model.add(Activation('relu'))
    
    model.add(Dropout(0.3))

    model.add(Dense(num_labels))
    model.add(Activation('softmax'))
    
    model.summary()
    return model

##### Setto parametri training #####
def setParamsTrain(model,learningrate):
    print("[INFO] compiling model...")
    adam = Adam(lr=learningrate)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])

##### Inizia il training #####
def startTrain(model,x_train,y_train,x_test,y_test,batch_size,epochs,modelname,tokenizer,tosavetokenizer):
    history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_split=0.1,
                    )

    ##### Fase di valutazione del modello #####
    score = model.evaluate(x_test, y_test, 
                       batch_size=batch_size, 
                       verbose=1
                       )

    print('Test accuracy:', score[1])
    ##### salvataggio del modello #####
    
    model.save(modelname)
    with open(tosavetokenizer, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

##### Predizioni del modello #####
def makePrediction(model,x_test,labelsTest,csvfile):
    print('Init prediction')
    ar=[
    "cronaca",
    "cultura",
    "economia",
    "esteri",
    "politica",
    "società",
    "sport",
    "tecnologia"
    ]   
    csv = open(csvfile, "w") 
    columnTitleRow = "LabelOriginale, LabelPredetta\n"
    csv.write(columnTitleRow)
    
    for i in range(len(x_test)):
        prediction = model.predict(np.array([x_test[i]])) #predizione sull'articolo
        predicted_label = ar[np.argmax(prediction[0])] #Restituisco l'indice con il valore massimo lungo un asse
        row = ar[labelsTest[i]] + "," + predicted_label+"\n"
        csv.write(row)




if __name__ == '__main__':
    dataset ='C:\\Users\\Samsung\\Desktop\\DataScience\\class\\ds_equal_classes.json'
    modelname = "C:\\Users\\Samsung\\Desktop\\model.h5"
    csvfile ="C:\\Users\\Samsung\\Desktop\\Prediction.csv"
    tosavetokenizer = 'C:\\Users\\Samsung\\Desktop\\tokenizer.pickle'
    num_labels = 8
    vocab_size = 15000
    batch_size = 16
    learningrate = 0.0001 
    epochs = 15
    mode = 'tfidf'
    data = dataReader()
    train,test = data.getJsonData(num_labels,dataset)
    obsTrain,labelsTrain = data.getValueDataset(train)
    obsTest,labelsTest = data.getValueDataset(test)
    x_train,x_test,y_train,y_test,tokenizer = makeTokenize(obsTrain,obsTest,labelsTrain,labelsTest,num_labels,vocab_size,mode)
    model = build_model(num_labels,vocab_size)
    setParamsTrain(model,learningrate)
    startTrain(model,x_train,y_train,x_test,y_test,batch_size,epochs,modelname,tokenizer,tosavetokenizer)
    makePrediction(model,x_test,labelsTest,csvfile)
