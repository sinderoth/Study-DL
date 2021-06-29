import tensorflow as tf
import matplotlib.pyplot as plt

class myCallBack(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('accuracy') > 0.95):
            self.model.stop_training = True




callbacks = myCallBack()
mnist = tf.keras.datasets.fashion_mnist

(training_images , training_labels) , (test_images , test_labels) = mnist.load_data()

#normalization
training_images  = training_images / 255.0
test_images = test_images / 255.0

model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(128,activation = tf.nn.relu),
                                    tf.keras.layers.Dense(10,activation = tf.nn.softmax)])

model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy',
metrics= ['accuracy'])

model.fit(training_images , training_labels , epochs = 50, callbacks = [callbacks])

model.evaluate(test_images , test_labels)

classifications = model.predict(test_images)
