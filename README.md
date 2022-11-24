**My Certification** : https://coursera.org/share/819d6cea1a99d62f875f347c05106167

## Useful Material
- Deeplearning-ai (Notes & Assignments)
https://github.com/https-deeplearning-ai/tensorflow-1-public
- Tensorflow Dataset 
https://www.tensorflow.org/datasets/catalog/overview
https://github.com/tensorflow/datasets/tree/master/docs/catalog
- NLP
https://projector.tensorflow.org/

## Course 1: Introduction to TensorFlow for AI, ML and DL

This first course introduces you to Tensor Flow, a popular machine learning framework. You will learn how to build a basic neural network for computer vision and use convolutions to improve your neural network.

#### Week 1: A New Programming Paradigm

- Introduction: A conversation with Andrew Ng
- A primer in machine learning
- The “Hello World” of neural networks
- Working through “Hello World” in TensorFlow and Python
- Week 1 - Predicting house price
  https://github.com/lau792301/Tensorflow-Developer-Professional-Practice/blob/main/1.%20Intro%20to%20Tensorflow/C1W1%20Housing%20Prices.ipynb

#### Week 2: Introduction to Computer Vision

- A conversation with Andrew Ng
- An introduction to computer vision
- Writing code to load training data
- Coding a computer vision neural network
- Walk through a notebook for computer vision
- Using callbacks to control training
- Walk through a notebook with callbacks
- Week 2 - Classifying Fashion MNIST with MLP
  https://github.com/lau792301/Tensorflow-Developer-Professional-Practice/blob/main/1.%20Intro%20to%20Tensorflow/C1W2%20Implementing%20Callbacks%20in%20TensorFlow%20using%20the%20MNIST%20Dataset.ipynb

#### Week 3: Enhancing Vision with Convolutional Neural Networks

- A conversation with Andrew Ng
- What are convolutions and pooling?
- Implementing convolutional layers
- Implementing pooling layers
- Improving the fashion classifier with convolutions
- Walking through convolutions
- Week 3 - Classifying Fashion MNIST with CNN
  https://github.com/lau792301/Tensorflow-Developer-Professional-Practice/blob/main/1.%20Intro%20to%20Tensorflow/C1W3%20Improve%20MNIST%20with%20Convolutions.ipynb

#### Week 4: Using Real-World Images

- A conversation with Andrew Ng
- Understanding ImageGenerator
- Defining a ConvNet to use complex images
- Training the ConvNet with fit_generator
- Walking through developing a ConvNet
- Walking through training the ConvNet with fit_generator
- Adding automatic validation to test accuracy
- Exploring the impact of compressing images
- Outro: Conversation with Andrew
- Week 4 - Classifying emotion with CNN
  https://github.com/lau792301/Tensorflow-Developer-Professional-Practice/blob/main/1.%20Intro%20to%20Tensorflow/C1W4%20Handling%20Complex%20Images%20-%20Happy%20or%20Sad%20Dataset.ipynb
## Course 2: Convolutional Neural Networks in TensorFlow

This second course teaches you advanced techniques to improve the computer vision model you built in Course 1. You will explore how to work with real-world images in different shapes and sizes, visualize the journey of an image through convolutions to understand how a computer “sees” information, plot loss and accuracy, and explore strategies to prevent overfitting, including augmentation and dropouts. Finally, Course 2 will introduce you to transfer learning and how learned features can be extracted from models.

#### Week 1: Exploring a Larger Dataset

- Introduction: A conversation with Andrew Ng
- Training with the cats vs. dogs dataset
- Working through the notebook
- Fixing through cropping
- Looking at accuracy and loss
- Week 1 - Classifying Cats and Dogs
  https://github.com/lau792301/Tensorflow-Developer-Professional-Practice/blob/main/2.%20CNN%20in%20Tensorflow/C2W1%20Using%20CNN's%20with%20the%20Cats%20vs%20Dogs%20Dataset.ipynb
  
#### Week 2: Augmentation, a Technique to Avoid Overfitting

- A conversation with Andrew Ng
- Introducing augmentation
- Coding augmentation with ImageDataGenerator
- Demonstrating overfitting in cats vs. dogs dataset
- Adding augmentation to cats vs. dogs dataset
- Exploring augmentation with horses vs. humans dataset
- Week 2 - Improving Cats and Dogs Classifier
  https://github.com/lau792301/Tensorflow-Developer-Professional-Practice/blob/main/2.%20CNN%20in%20Tensorflow/C2W2%20Tackle%20Overfitting%20with%20Data%20Augmentation.ipynb

#### Week 3: Transfer Learning

- A conversation with Andrew Ng
- Understanding transfer learning: the concepts
- Coding your own model with transferred features
- Exploring dropouts
- Exploring transfer learning with inception
- Week 3 - Transfer learning (VGG Net)
  https://github.com/lau792301/Tensorflow-Developer-Professional-Practice/blob/main/2.%20CNN%20in%20Tensorflow/C2W3%20Transfer%20Learning.ipynb

#### Week 4: Multi-class Classifications

- A conversation with Andrew Ng
- Moving from binary to multi-class classification
- Exploring multi-class classification with the rock paper scissors dataset
- Training a classifier with the rock paper scissors dataset
- Testing the rock paper scissors classifier
- Week 4 - Classifying images of sign languages
  https://github.com/lau792301/Tensorflow-Developer-Professional-Practice/blob/main/2.%20CNN%20in%20Tensorflow/C2W4%20Multi-class%20Classification.ipynb

## Course 3: Natural Language Processing in TensorFlow

In this third course, you’ll learn how to apply neural networks to solve natural language processing problems using TensorFlow. You’ll learn how to process and represent text through tokenization so that it’s recognizable by a neural network. You’ll be introduced to new types of neural networks, including RNNs, GRUs and LSTMs, and how you can train them to understand the meaning of text. Finally, you’ll learn how to train LSTMs on existing text to create original poetry and more!

#### Week 1: Sentiment in Text

- Introduction: A conversation with Andrew Ng
- Word-based encodings
- Using APIs
- Text to sequence
- Sarcasm, really?
- Working with the Tokenizer
- Week 1.1 - Detecting sarcasm in news headlines with LSTM and CNN
- Week 1.2 - Exploring BBC news data
  https://github.com/lau792301/Tensorflow-Developer-Professional-Practice/blob/main/3.%20Natural%20Language%20Processing%20in%20TensorFlow/C3W1%20Explore%20the%20BBC%20News%20archive.ipynb

#### Week 2: Word Embeddings

- A conversation with Andrew Ng
- The IMDB dataset
- Looking into the details
- How can we use vectors?
- More into the details
- Remember the sarcasm dataset?
- Building a classifier for the sarcasm dataset
- Let’s talk about the loss function
- Pre-tokenized datasets
- Diving into the code
- Week 2.1 - Classifying IMDB reviews data (Embedding + MLP)
- Week 2.2 - Classifying BBC news into topics (Embedding + Conv + MLP)
  https://github.com/lau792301/Tensorflow-Developer-Professional-Practice/blob/main/3.%20Natural%20Language%20Processing%20in%20TensorFlow/C3W2%20Diving%20deeper%20into%20the%20BBC%20News%20archive.ipynb

#### Week 3: Sequence Models

- A conversation with Andrew Ng
- LSTMs
- Implementing LSTMs in code
- A word from Laurence
- Accuracy and Loss
- Using a convolutional network
- Going back to the IMDB dataset
- Tips from Laurence
- Week 3.1 - Classifying IMDB reviews (Embedding + Conv1D)
- Week 3.2 - Twitter sentiment classification (GloVe)
  https://github.com/lau792301/Tensorflow-Developer-Professional-Practice/blob/main/3.%20Natural%20Language%20Processing%20in%20TensorFlow/C3W3%20Exploring%20Overfitting%20in%20NLP.ipynb

#### Week 4: Sequence Models and Literature

- A conversation with Andrew Ng
- Training the data
- Finding what the next word should be
- Predicting a word
- Poetry!
- Laurence the poet
- Week 4 - Poem generation with Bi-directional LSTM
  https://github.com/lau792301/Tensorflow-Developer-Professional-Practice/blob/main/3.%20Natural%20Language%20Processing%20in%20TensorFlow/C3W4%20Predicting%20the%20next%20word.ipynb

## Course 4: Sequences, Time Series, and Prediction

In this fourth course, you will learn how to solve time series and forecasting problems in TensorFlow. You’ll first implement best practices to prepare data for time series learning. You’ll also explore how RNNs and ConvNets can be used for predictions. Finally, you’ll apply everything you’ve learned throughout the Specialization to build a sunspot prediction model using real-world data!

#### Week 1: Sequences and Prediction

- Introduction: a conversation with Andrew Ng
- Time series examples
- Machine learning applied to time series
- Common patterns in time series
- Introduction to time series
- Train, validation, and test sets
- Metrics for evaluating performance
- Moving average and differencing
- Trailing versus centered windows
- Forecasting
- Week 1 - Create and predict synthetic data with time series decomposition
  https://github.com/lau792301/Tensorflow-Developer-Professional-Practice/blob/main/4.%20Sequences%20and%20Prediction/C4W1%20Working%20with%20time%20series.ipynb

#### Week 2: Deep Neural Networks for Time Series

- A conversation with Andrew Ng
- Preparing features and labels
- Feeding a windowed dataset into a neural network
- Single layer neural network
- Machine learning on time windows
- Prediction
- More on single-layer network
- Deep neural network training, tuning, and prediction
- Week 2.1 - Prepare features and labels
- Week 2.2 - Predict synthetic data with Linear Regression
- Week 2.3 - Predict synthetic data with MLP
https://github.com/lau792301/Tensorflow-Developer-Professional-Practice/blob/main/4.%20Sequences%20and%20Prediction/C4W2%20Predicting%20time%20series.ipynb

#### Week 3: Recurrent Neural Networks for Time Series

- A conversation with Andrew Ng
- Shape of the inputs to the RNN
- Outputting a sequence
- Lambda layers
- Adjusting the learning rate dynamically
- RNNs
- LSTMs
- Coding LSTMs
- More on LSTMs
- Week 3.1 - Finding an optimal learning rate for a RNN
- Week 3.2 - LSTM
https://github.com/lau792301/Tensorflow-Developer-Professional-Practice/blob/main/4.%20Sequences%20and%20Prediction/C4W3%20Using%20RNNs%20to%20predict%20time%20series.ipynb

#### Week 4: Real-world Time Series Data

- A conversation with Andrew Ng
- Convolutions
- Bi-directional LSTMs
- Real data – sunspots
- Train and tune the model
- Prediction
- Sunspots
- Combining our tools for analysis
- Week 4 - Apply into real world data
https://github.com/lau792301/Tensorflow-Developer-Professional-Practice/blob/main/4.%20Sequences%20and%20Prediction/C4W4%20Using%20real%20world%20data.ipynb
