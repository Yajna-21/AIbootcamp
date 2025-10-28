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
#plt.imshow(x_train[0])
#plt.show()
print("**************")
print(f"label is : {y_train[0]}")

# normalize
x_train=x_train.astype('float32')/255.0
x_test=x_test.astype('float32')/255.0

#to_categorical
print(f"before:label is:{y_train[100]}")
y_train = to_categorical(y_train)
print(f"after:label is:{y_train[100]}")
y_test=to_categorical(y_test)

#architecture
model=Sequential()
model.add(Flatten(input_shape=(28,28)))
model.add(Dense(128,activation='relu'))  #128 neurons
model.add(Dense(10,'softmax'))

#compile the model
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

#train
result=model.fit(x_train,y_train,epochs=10,batch_size=64,validation_split=0.2) #64 images

# evaluate
loss,accuracy=model.evaluate(x_test,y_test)
print(f"test loss:{loss}")
print(f"test accuracy:{accuracy}")
print(result.history.keys())
print(result.history.values())
print(result.history)

print('##########')

#visualization
plt.plot(result.history["accuracy"],label="Training Accuracy",color='blue')
plt.plot(result.history['loss'],label="Training Loss",color='orange')
plt.title("Training accuracy vs Val_accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

