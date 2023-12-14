import tensorflow as tf
mnist = tf.keras.datasets.mnist
(x_train,y_train),(x_test, y_test) = mnist.load_data()#importing mnist datasets into x_train...

#normalize pixel values to between 0 and 1
x_train,x_test = x_train/255.0, x_test/255.0

#Designing NN architecture

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)), #Creates the first layer of input.
    tf.keras.layers.Dense(128,activation = 'relu'),#Creates connected layer of 128 neurons and applies ReLU function
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10,activation='softmax')#Creates third layer of digits from (0-9) and applies softmax algorithm
    ])

model.compile(optimizer = 'adam',loss = 'sparse_categorical_crossentropy',metrics = ['accuracy'])

model.fit(x_train,y_train,epochs = 3)

test_loss,test_acc = model.evaluate(x_test,y_test)
print('Test accuracy:',test_acc)

model.save('nnh.h5')
print('Model saved')
