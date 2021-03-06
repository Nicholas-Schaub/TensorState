import os

# Set the log level to hide some basic warning/info generated by Tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Fix for cudnn error on RTX gpus
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import tensorflow as tf
import tensorflow.keras as keras
import TensorState as ts
import numpy as np
import time

""" Load MNIST and transform it """
# Load the data
mnist = keras.datasets.mnist
(train_images,train_labels), (test_images,test_labels) = mnist.load_data()

# Normalize the data
train_images = train_images/255
test_images = test_images/255

# Add a channel axis
train_images = train_images[..., tf.newaxis]
test_images = test_images[..., tf.newaxis]

""" Create a LeNet-5 model """
# Set the random seed for reproducibility
tf.random.set_seed(0)

# Set the convolutional layer settings
reg = keras.regularizers.l2(0.0005)
kwargs = {'activation': 'elu',
          'kernel_initializer': 'he_normal',
          'kernel_regularizer': reg,
          'bias_regularizer': reg}

# Build the layers
input_layer = keras.layers.Input(shape=(28,28,1), name='input')

# Unit 1
conv_1 = keras.layers.Conv2D(20, 5, name='conv_1',**kwargs)(input_layer)
norm_1 = keras.layers.BatchNormalization(epsilon=0.00001,momentum=0.9)(conv_1)
maxp_1 = keras.layers.MaxPool2D((2,2), name='maxp_1')(norm_1)

# Unit 2
conv_2 = keras.layers.Conv2D(50, 5, name='conv_2', **kwargs)(maxp_1)
norm_2 = keras.layers.BatchNormalization(epsilon=0.00001,momentum=0.9)(conv_2)
maxp_2 = keras.layers.MaxPool2D((2,2), name='maxp_2')(norm_2)

# Fully Connected
conv_3 = keras.layers.Conv2D(100, 4, name='conv_3', **kwargs)(maxp_2)
norm_3 = keras.layers.BatchNormalization(epsilon=0.00001,momentum=0.9)(conv_3)

# Prediction
flatten = keras.layers.Flatten(name='flatten')(norm_3)
pred = keras.layers.Dense(10,name='pred')(flatten)

# Create the Keras model
model = keras.Model(
                    inputs=input_layer,
                    outputs=pred
                   )

print(model.summary())

""" Train the model """
# Compile for training
model.compile(
              optimizer=keras.optimizers.SGD(learning_rate=0.001,momentum=0.9,nesterov=True),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,name='loss'),
              metrics=['accuracy']
             )

# Stop the model once the validation accuracy stops going down
earlystop_callback = tf.keras.callbacks.EarlyStopping(
                            monitor='val_accuracy',
                            mode='max',
                            patience=5,
                            restore_best_weights=True
                        )

# Train the model
model.fit(
          train_images, train_labels, epochs=200, 
          validation_data=(test_images, test_labels),
          batch_size=200,
          callbacks=[earlystop_callback],
          verbose=1
         )

""" Evaluate model efficiency """
# Attach StateCapture layers to the model
efficiency_model = ts.build_efficiency_model(model,attach_to=['Conv2D'],method='after')

# Collect the states for each layer
print()
print('Running model predictions to capture states...')
start = time.time()
predictions = efficiency_model.predict(test_images,batch_size=200)
print('Finished in {:.3f}s!'.format(time.time() - start))

# Count the number of states in each layer
print()
print('Getting the number of states in each layer...')
for layer in efficiency_model.efficiency_layers:
    print('Layer {} number of states: {}'.format(layer.name,layer.state_count))

# Calculate each layers efficiency
print()
print('Evaluating efficiency of each layer...')
for layer in efficiency_model.efficiency_layers:
    start = time.time()
    print('Layer {} efficiency: {:.1f}% ({:.3f}s)'.format(layer.name,100*layer.efficiency(),time.time() - start))

# Calculate the aIQ
beta = 2 # fudge factor giving a slight bias toward accuracy over efficiency

print()
print('Network metrics...')
print('Beta: {}'.format(beta))

network_efficiency = ts.network_efficiency(efficiency_model)
print('Network efficiency: {:.1f}%'.format(100*network_efficiency))

accuracy = np.sum(np.argmax(predictions,axis=1)==test_labels)/test_labels.size
print('Network accuracy: {:.1f}%'.format(100*accuracy))

aIQ  = ts.aIQ(network_efficiency,accuracy,beta)
print('aIQ: {:.1f}%'.format(100*aIQ))