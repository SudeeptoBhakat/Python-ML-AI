# Real Time Facial Emotion Detection & ML Playground

A practical computer vision and deep learning project for real time facial emotion recognition using a Convolutional Neural Network (CNN) and OpenCV Haar Cascades, alongside NLP experiments for spam text classification.

---

## 1. What This Project Does

The core of this repository is a **real-time facial emotion recognition pipeline**:
1. It takes live video from your webcam.
2. Uses OpenCV's Haar Cascade classifier (`haarcascade_frontalface_default.xml`) to detect faces in each frame.
3. Crops and resizes each detected face to a `48x48` grayscale image.
4. Passes the processed face through a custom-trained CNN model (`emotion_cnn_model.h5` / `emotion_detection_model.h5`).
5. Predicts one of **5 emotion classes** and draws a bounding box with the emotion tag directly over the video feed:
   - `0`: Angry
   - `1`: Happy
   - `2`: Neutral
   - `3`: Sad
   - `4`: Surprised

In addition to the emotion detection pipeline, the repository includes NLP text classification notebooks (SMS spam detection with TF-IDF and Logistic Regression) and foundational OpenCV/NLP experimentation sheets.

---

## 2. Directory & File Breakdown

| File / Directory | Description |
| :--- | :--- |
| `emotion_dection/` | Core emotion recognition project directory. |
| `emotion_dection/Emotion/` | Raw facial image dataset split into 5 class subfolders (`angry/`, `happy/`, `neutral/`, `sad/`, `surprised/`) containing ~16,000+ samples. |
| `emotion_dection/data/` | Serialized pickled data (`images.p` and `labels.p`) for fast training without reading thousands of loose image files repeatedly. |
| `emotion_dection/emotion_detection_model.h5` | Trained Keras, TensorFlow HDF5 model weights. |
| `emotion_dection/demo.ipynb` | Dataset preprocessing, image array conversions, and CNN training notebook. |
| `EMOTIONDETECTIONMODELTRAINING.ipynb` | GPU training notebook for the CNN model. |
| `Detection.ipynb` | Real-time webcam inference script with OpenCV face crop & CNN prediction overlay. |
| `haarcascade_frontalface_default.xml` | Pre-trained OpenCV Haar Cascade model for frontal face detection. |
| `CNN Model Training emotion Detection.pdf` | Documented workflow notes for model training. |
| `Emotion Pickling Note.pdf` | Notes on image array serialization and pickling steps. |
| `notes/` | Handwritten study notes and reference diagrams for the pipeline. |
| `cleaned_dataset*.csv` | Cleaned SMS dataset for spam classification (NLP practice). |
| `Untitled-1.ipynb` | NLP Spam Classifier notebook using TF-IDF vectorization and Logistic Regression. |
| `Untitled3.ipynb` | OpenCV computer vision image reading and manipulation practice notebook. |
| `Untitled4.ipynb` | NLP study guide covering tokenization, stemming, lemmatization, and POS tagging with NLTK. |
| `8484902-hd_1920_1080_25fps.mp4` | Sample test video for offline face & emotion detection testing. |

---

### A. Data Preprocessing & Pickling
1. Raw image files in `emotion_dection/Emotion/` are read, converted to grayscale, and resized to `(48, 48)`.
2. Image pixels are stored into a NumPy array `x` and string labels (`angry`, `happy`, etc.) into `y`.
3. The datasets are dumped as binary files (`images.p`, `labels.p`) using Python `pickle` to avoid filesystem overhead during iterative training runs.

### B. CNN Architecture
The model is built using Keras `Sequential`:
- **Conv2D (28/45 filters, 3x3, ReLU)** + **MaxPooling2D (2x2)**
- **Conv2D (56/64 filters, 3x3, ReLU)** + **MaxPooling2D (2x2)**
- **Conv2D (112/128 filters, 3x3, ReLU)** + **MaxPooling2D (2x2)**
- **Flatten** (converts 3D feature maps to 1D feature vector)
- **Dense (256 units, ReLU)**
- **Dense (5 units, Softmax)** (output probabilities across 5 emotions)
- **Loss**: `categorical_crossentropy` | **Optimizer**: `Adam(lr=0.001)`

### C. Live Inference
- Frame captured by `cv2.VideoCapture(0)`.
- Grayscale converted frame is scanned using `face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)`.
- For every detected face:
  ```python
  face = gray[y:y+h, x:x+w]
  face = cv2.resize(face, (48, 48)) / 255.0
  face = face.reshape(1, 48, 48, 1)
  prediction = model.predict(face, verbose=0)
  emotion = emotion_labels[np.argmax(prediction)]
  ```
- Resulting emotion string is drawn via `cv2.putText` alongside `cv2.rectangle`.


## 3. Additional Practice Modules

- **SMS Spam Detection (`Untitled-1.ipynb`)**:
  Cleans SMS text messages, strips noise, generates TF-IDF vectors, and trains a Logistic Regression classifier to differentiate between `ham` (legitimate) and `spam` messages.
- **NLP Foundations (`Untitled4.ipynb`)**:
  Walkthrough of standard text normalization pipelines using NLTK (Tokenization, Stopwords, Stemming vs. Lemmatization, POS tagging, and Named Entity Recognition).
