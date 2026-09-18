# Python-ML-AI
This repository features Python-based Machine Learning (ML) and Artificial Intelligence (AI) projects.

## Real-Time Facial Emotion Detection
A deep learning project using a Convolutional Neural Network (CNN) built with Keras/TensorFlow and OpenCV Haar Cascades (`haarcascade_frontalface_default.xml`). It detects faces from live webcam feeds or video files, crops and normalizes the facial regions to 48x48 grayscale inputs, and classifies them across 5 emotion categories (Angry, Happy, Neutral, Sad, Surprised) in real time.

## Image Classification Model
A model for classifying images using OpenCV for face and eye detection, wavelet transforms for feature extraction, and machine learning algorithms (SVM, Random Forest, Logistic Regression) for classification.

## SMS Spam Classifier (NLP)
A natural language processing pipeline for text classification. It cleans and tokenizes raw SMS messages, extracts feature vectors using TF-IDF vectorization, and trains a Logistic Regression classifier to distinguish between spam and legitimate (ham) messages.

## Content Generator
This repo includes a `content_generator` function that uses a language model endpoint from Hugging Face to generate responses based on user queries. The function leverages few-shot learning with example-based prompting to enhance response accuracy.

## NLP Foundations & Text Mining
Exploratory notebooks and study guides covering foundational text preprocessing techniques using NLTK—including tokenization, stopword removal, stemming, lemmatization, and POS tagging.

---

The repository also includes sample datasets, pickled feature arrays for fast training workflows, and interface setups (including Django frontend integration and interactive Jupyter notebooks) for seamless experimentation.
