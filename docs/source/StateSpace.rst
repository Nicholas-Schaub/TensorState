====================
What is state space?
====================

-----------------------
An Abstract Explanation
-----------------------

Traditionally, the fundamental unit of computation in the human nervous system
has been the neuron. Some of the early thought in how the brain processes
information suggested that particular neurons encode particular information,
which may be called features in the data science community. Thus, if we have
~86 billion neurons, then we can roughly learn 86 billion features. However, as
human intelligence progressed it was quickly discovered that even simple tasks
often recruit multiple, seemingly unrelated areas of the brain. This led to the
idea that the meaning of an individual neuron may not matter as much as how
neurons fire as a collective, suggesting the number of neurons are an exponent
rather than a coefficient to a human's capacity to learn new tasks.

Recent research into artificial neural networks and how they function approach
the subject in the antiquated approach to understanding human neurons, where
each neuron encodes a particular feature or function. The idea of state space
diverges and challenges this conception of neurons, and attempts to show how
neurons in artificial neural networks operate in parallel rather than discrete
units in the same way that the human brain operates. In this conception of a
neuron layer, the features are encoded in the state of firing neurons rather
than individual neurons. Thus, in the current conception of neural networks, a
layer with 16 neurons would encode 16 features, but in the state space up to
2^16 features can be encoded (assuming neurons are either firing or not firing).