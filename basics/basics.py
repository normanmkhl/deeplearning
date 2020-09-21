import tensorflow as tf
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist # 28 x 28 images of hand-written digits 0-9

(x_train, y_train), (x_test, y_test) = mnist.load_data() # data unpack

x_train = tf.keras.utils.normalize(x_train, axis = 1) # 0 - 1 range
x_test = tf.keras.utils.normalize(x_test, axis = 1) 

model = tf.keras.models.Sequential() # sequential model
model.add(tf.keras.layers.Flatten()) # single dimension
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu)) # first layer
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax)) # output layer

model.compile(optimizer ='adam', 
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']) # model parameter training

model.fit(x_train, y_train, epochs=3) # model training , epoch = pass through data set 3 times

val_loss, val_acc = model.evaluate(x_test, y_test)
print(val_loss, val_acc) # loss & accuracy

plt.imshow(x_train[0], cmap = plt.cm.binary) # color map binary
plt.show()
print(x_train[0]) 

model.save('test.model')

new_model = tf.keras.models.load_model('test.model')

predictions = new_model.predict(x_test) # prediction (probability distribution)
print(predictions)

import numpy as np

print(np.argmax(predictions[0])) # 0 index x_test prediction
plt.imshow(x_test[0])
plt.show()