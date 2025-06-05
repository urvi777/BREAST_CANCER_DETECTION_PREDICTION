# 🧠 Breast Cancer Classification with Neural Network (NN)

This project leverages a simple **Neural Network** model built using TensorFlow and Keras to classify breast cancer tumors as **malignant** or **benign** based on diagnostic measurements.

---

## 📌 Project Overview

- **Objective**: Classify whether a tumor is malignant or benign.
- **Dataset**: Breast cancer dataset from `sklearn.datasets`.
- **Frameworks**: `TensorFlow`, `Keras`, `NumPy`, `Matplotlib`, `Pandas`, `Scikit-learn`.

---

## 📊 Dataset Details

The dataset contains features computed from digitized images of a fine needle aspirate (FNA) of a breast mass. These features describe characteristics of the cell nuclei present in the image.

- **Total Samples**: 569
- **Features**: 30 numeric features (like radius, texture, smoothness, etc.)
- **Target**:
  - `0` → Malignant (cancerous)
  - `1` → Benign (non-cancerous)

---

## 🧼 Data Preprocessing

- Loaded dataset using `sklearn.datasets.load_breast_cancer()`.
- Converted it into a pandas `DataFrame`.
- Added target labels to the DataFrame.
- Checked for missing values and confirmed clean data.
- Split the data into training and testing sets (80/20).
- Standardized feature data using `StandardScaler`.

---

## 🧠 Model Architecture

Using `TensorFlow` and `Keras`, a sequential neural network model was created:

```python
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(30,)),
    keras.layers.Dense(20, activation='relu'),
    keras.layers.Dense(2, activation='sigmoid')
])
Input Layer: Flatten input of shape (30,)
Hidden Layer: Dense with 20 neurons and ReLU activation
Output Layer: Dense with 2 neurons and sigmoid activation for binary classification

⚙️ Model Compilation & Training
Loss Function: sparse_categorical_crossentropy
Optimizer: adam
Metrics: accuracy
Training Epochs: 10
Validation Split: 10% from training data

📈 Model Evaluation
Visualized the model's accuracy and loss:
Training vs Validation Accuracy
Training vs Validation Loss


🔮 Making Predictions
Used model.predict() to get class probabilities.
Converted probabilities to class labels using np.argmax().
Built a simple predictive system using a manual input sample:

🗂️ Files Included
breast_cancer_classification_with_nn.py: Python code implementing the entire project.
README.md: Project documentation (this file).

🚀 Future Improvements
Add early stopping and dropout for better generalization.
Test different optimizers and deeper neural network structures.
Deploy the model using a web app (Streamlit or Flask).

👩‍💻 Author
Urvi Patel
Machine Learning Enthusiast | Intern

