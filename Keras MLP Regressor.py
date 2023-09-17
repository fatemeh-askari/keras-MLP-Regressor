#keras MLP Regressor , 2023-02-25

# Steps
## Replace
### Additional information

# import library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import openpyxl

# 1) import dataset
data = pd.read_csv( ##'F:/Machin Learning/Book1convnet.csv')
### data.head
      
# 2) inputs and outputs
x = data [##['Open', 'High', 'Low']]
y = data [##'Close']
    
# 3) Split dataset into training set and test set
trainX, testX, trainY, testY = train_test_split(x, y, test_size = ## 0.2)
                                                
# 4) Scaling                                               
sc=StandardScaler()

scaler = sc.fit(trainX)
trainX_scaled = scaler.transform(trainX)
testX_scaled = scaler.transform(testX)


# 5) Hyper parameters and network architecture and fit model in keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation
Model = Sequential()
Model.add(Dense(## 3, input_dim=3))
Model.add(Activation(## 'relu'))
Model.add(Dense(## 5))
Model.add(Activation(## 'relu'))
Model.add(Dense(## 1))

from keras.optimizers import ## Adadelta
Model.compile(loss='mse', optimizer = ## Adadelta(learning_rate = ##0.4))

history_train_model = Model.fit(trainX_scaled, trainY, epochs = ## 8, batch_size=50)


# 6) loss in traning and plot it
print(history_train_model.history['loss'])
plt.plot(history_train_model.history['loss'])
plt.title('the loss of traning model')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show

# 7) test 
predictions = Model.predict(testX_scaled)
#print(predictions)
loss = Model.evaluate (testX_scaled, testY)
print(loss)









