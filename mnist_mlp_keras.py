
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.datasets import mnist
from keras.utils import to_categorical
import matplotlib.pyplot as plt

# load the data
(x_train,y_train),(x_test,y_test)=mnist.load_data()
print(x_train.dtype)
print(x_train.shape)
print(y_test.shape)
print(x_train[0])
plt.imshow(x_train[0])
plt.show()
print("**************")
print(f"label is : {y_train[0]}")

# normalize
x_train=x_train.astype('float32')/255.0
x_test=x_test.astype('float32')/255.0

#to_categorical
print(f"before : label is : {y_train[0]}")
y_test=to_categorical(y_test)
#architecture
model=Sequential()
model.add(Flatten(input_shape=(28,28)))
model.add(Dense(128,activation='relu'))
model.add(Dense(10,activation='softmax'))
#compile
model.compile(optimizer='Adam',loss='categorical_crossentropy')
#train
model.fit(x_train,y_train,epochs=5,batch_size=64)
print("#####%%######")
