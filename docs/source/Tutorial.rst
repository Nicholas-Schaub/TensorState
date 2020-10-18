========
Tutorial
========

------------
Introduction
------------

The core ideas behind this package were originally described in our paper,
`Assessing Intelligence in Artificial Neural Networks <https://arxiv.org/abs/2006.02909>`_.

The fundamental concept behind the work is that the power of a neural network
is found in the combination of firing and non-firing neurons rather than individual
neurons. For a given input, a state is the combination of firing and none-firing
neurons. To make computation easier and more storage friendly, neuron activation
is quantized into on/off states based on whether the neurons value is ``>0`` (on) or
``<=0`` (off).

If information is stored in the combination of firing neurons, rather than
into discrete neurons, then the information storage space of neural networks
is considerably larger than conventionally supposed. For example, a neuron 
layer with 64 neurons may be considered to have 64 unique features by convential
thinking. However, if information is stored in the state space, then the number
of features may be up to 2^64. This conception of how information is stored
in the network may explain why neural networks are capable of `memorizing labels
on large data sets <https://arxiv.org/abs/1611.03530>`_.

This package simplifies the capture of neural layers states, and provides some
utility functions to assist in analyzing the state space of neural layers.

------------------------------------------------
Attach StateCapture Layers to a Tensorflow Model
------------------------------------------------

In this tutorial, we are going to build a classic convolutional neural network, LeNet-5.
Then we are going to use TensorState to evaluate the network architecture, getting the
efficiency of each layer and calculating the artificial intelligence quotient.

'''''''''
Get MNIST
'''''''''

To train this model, we need to get the MNIST data set. Fortunately, it comes packaged
with Keras in Tensorflow. The original data was 8-bit and a single channel, so we need
to add a channel axis and we are going to normalize the image to have pixel values
ranging from 0-1.

.. code-block:: python

    import tensorflow.keras as keras

    # Load the data
    mnist = keras.datasets.mnist
    (train_images,train_labels), (test_images,test_labels) = mnist.load_data()

    # Normalize the data
    train_images = train_images/255
    test_images = test_images/255

    # Add a channel axis
    train_images = train_images[..., tf.newaxis]
    test_images = test_images[..., tf.newaxis]

''''''''''''''''''''''
Create a LeNet-5 Model
''''''''''''''''''''''

The general structure of our LeNet-5 model will roughly follow the structure of the
original network described by
`LeCun et al. <http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf>`_
However, we are going to make a few modifications to make it more modern relative to
the original architecture, such as addition of l2 weight regularization, exponential
linear units, and batch normalization. So, we start off by setting the Tensorflow
random seed (to make the results reproducible) and set up the parameters of the
convolutional layers.

.. code-block:: python

    import tensorflow as tf

    # Set the random seed for reproducibility
    tf.random.set_seed(0)

    # Set the convolutional layer settings
    reg = keras.regularizers.l2(0.0005)
    kwargs = {'activation': 'elu',
              'kernel_initializer': 'he_normal',
              'kernel_regularizer': reg,
              'bias_regularizer': reg}

Next, we create the layers of the network. LeNet-5 has 2 convolutional layers and 2
fully connected layers. We will use max pooling layers after each convolutional layer,
and we will add a batch normalization layer to all by the last fully connected layer.

.. code-block:: python

    # Build the layers
    input_layer = keras.layers.Input(shape=(28,28,1), name='input')

    # Unit 1
    conv_1 = keras.layers.Conv2D(20, 5, name='conv_1',**kwargs)(input_layer)
    norm_1 = keras.layers.BatchNormalization(epsilon=0.00001,momentum=0.9)
    maxp_1 = keras.layers.MaxPool2D((2,2), name='maxp_1')(conv_1)

    # Unit 2
    conv_2 = keras.layers.Conv2D(50, 5, name='conv_2', **kwargs)(maxp_1)
    norm_2 = keras.layers.BatchNormalization(epsilon=0.00001,momentum=0.9)
    maxp_2 = keras.layers.MaxPool2D((2,2), name='maxp_2')(conv_2)

    # Fully Connected (even a convolutional layer is used)
    conv_3 = keras.layers.Conv2D(100, 4, name='conv_3', **kwargs)(maxp_2)
    norm_3 = keras.layers.BatchNormalization(epsilon=0.00001,momentum=0.9)

    # Fully Connected (Prediction)
    flatten = keras.layers.Flatten(name='flatten')(conv_3)
    pred = keras.layers.Dense(10,name='pred')(flatten)

    # Create the Keras model
    model = keras.Model(
                        inputs=input_layer,
                        outputs=pred
                       )

'''''''''''''''''''''''
Train the LeNet-5 Model
'''''''''''''''''''''''

Next we train the LeNet-5 model, and stopping as soon as the validation accuracy stops
increasing.

.. code-block:: python
    # Compile for training
    model.compile(
                optimizer=keras.optimizers.SGD(lr=0.001,momentum=0.9,nesterov=True),
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

'''''''''''''''''''''''''''''''''''
Use TensorState to Evaluate LeNet-5
'''''''''''''''''''''''''''''''''''

To calculate neural layer efficiency, we need to capture the various states each layer
takes on as the network processes data. This functionality is built into the
``StateCapture`` class, which is a Tensorflow layer that can be inserted into the model
to automate the capturing of information passing through the network. The
``StateCapture`` layer acts like a probe that can be placed anywhere in the network:
it records the information without modifying it, and passes it on to subsequent layers.

While ``StateCapture`` layers can be placed manually, there is a convenience function
that can take an existing neural network and return a new network with ``StateCapture``
layers inserted at the designated areas. For example, we can attach a ``StateCapture``
layer to all convolutional layers.

.. code-block:: python

    import TensorState as ts
    efficiency_model = ts.build_efficiency_model(model,attach_to=['Conv2D'],method='after')

In the above code, we feed the trained LeNet-5 model into the function, designate we want
to attach ``StateCapture`` layers to all 2D convolutional layers, and we want to capture
the states ``after`` the layer. We could also capture the inputs going into and out of the
layer by using ``method='both'``. For more information on the ``build_efficiency_model``
method and additional settings, please see the TensorState reference.

Now that the ``efficiency_model`` has been created, the ``StateCapture`` layers will
collect all states of the network as images are fed to the network. Thus, to generate all
possible states the network contains for the test data, we only need to ``predict`` the
classes for the test data. Then we can look at how many states were collected for each
layer.

.. code-block:: python

    predictions = efficiency_model.predict(train_images,batch_size=200)

    for layer in efficiency_model.efficiency_layers:
        print('Layer {} number of states: {}'.format(layer.name,layer.state_count))
    
Note how ``efficiency_model`` has the efficiency layers stored in the ``efficiency_layers``
attribute of the model. The output of the above code should look something like this:

.. code-block:: bash
    
    Layer conv_1_states number of states: 34560000
    Layer conv_2_states number of states: 3840000
    Layer conv_3_states number of states: 60000

Since there are 60,000 images in the training data set, it is expected that the fully
connected layer (``conv_3_states``) has 60,000 states recorded, since exactly one state
will be recorded per image. The other layers are convolutional, generating multiple
states per image. The number of states can be checked by determining the number of
locations the convolutional operator is applied per image then multiplying by 60,000.
For example, in a 28x28 image with a 5x5 convolutional operation performed on it, the
dimensions of the output would be 24x24. Thus, the number of states for all 60,000 images
would be 24*24*60,000=34,560,000 states, which is the number of states observed by
``conv_1_states``.

.. note::

    The ``state_count`` is the raw number of states observed, and there are likely states
    that occur multiple times.

Now that the states of each layer have been captured, let's analyze the state space using
the efficiency metric originally described by
`Schaub et al <https://arxiv.org/abs/2006.02909>`_. The efficiency metric calculates the
entropy of the state space and divides by the number of neurons in the layer, giving an
efficiency value in the range 0.00-1.00.

.. code-block:: python

    for layer in efficiency_model.efficiency_layers:
        layer_efficiency = layer.efficiency()
        print('Layer {} efficiency: {:.1f}%'.format(layer.name,100*layer_efficiency))

Next, we can calculate the artificial intelligence quotient (aIQ). Since things like
neural network efficiency and aIQ are metrics calculated over the entire network,
the ``StateCapture`` layer does not have built-in methods to calculate these values.

.. code-block:: python

    beta = 2 # fudge factor giving a slight bias toward accuracy over efficiency

    print()
    print('Network metrics...')
    print('Beta: {}'.format(beta))

    network_efficiency = ts.network_efficiency(efficiency_model)
    print('Network efficiency: {:.1f}%'.format(100*network_efficiency))

    accuracy = np.sum(np.argmax(predictions,axis=1)==train_labels)/train_labels.size
    print('Network accuracy: {:.1f}%'.format(100*accuracy))

    aIQ  = ts.aIQ(network_efficiency,accuracy,beta)
    print('aIQ: {:.1f}%'.format(100*aIQ))

----------------
Complete Example
----------------

.. code-block:: python

    # Set the log level to hide some basic warning/info generated by Tensorflow
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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
    norm_1 = keras.layers.BatchNormalization(epsilon=0.00001,momentum=0.9)
    maxp_1 = keras.layers.MaxPool2D((2,2), name='maxp_1')(conv_1)

    # Unit 2
    conv_2 = keras.layers.Conv2D(50, 5, name='conv_2', **kwargs)(maxp_1)
    norm_2 = keras.layers.BatchNormalization(epsilon=0.00001,momentum=0.9)
    maxp_2 = keras.layers.MaxPool2D((2,2), name='maxp_2')(conv_2)

    # Fully Connected
    conv_3 = keras.layers.Conv2D(100, 4, name='conv_3', **kwargs)(maxp_2)
    norm_3 = keras.layers.BatchNormalization(epsilon=0.00001,momentum=0.9)

    # Prediction
    flatten = keras.layers.Flatten(name='flatten')(conv_3)
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
                  optimizer=keras.optimizers.SGD(lr=0.001,momentum=0.9,nesterov=True),
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
    predictions = efficiency_model.predict(train_images,batch_size=200)
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

    accuracy = np.sum(np.argmax(predictions,axis=1)==train_labels)/train_labels.size
    print('Network accuracy: {:.1f}%'.format(100*accuracy))

    aIQ  = ts.aIQ(network_efficiency,accuracy,beta)
    print('aIQ: {:.1f}%'.format(100*aIQ))