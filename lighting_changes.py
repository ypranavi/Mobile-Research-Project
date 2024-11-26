import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

# 1. Load and Preprocess the SoF Dataset
def load_images(dataset_path, target_size=(128, 128)):
    """Loads images and labels from the SoF dataset directory."""
    images, labels = [], []
    for label in os.listdir(dataset_path):  # Each subdirectory is a label
        label_path = os.path.join(dataset_path, label)
        if os.path.isdir(label_path):
            for img_file in os.listdir(label_path):
                img_path = os.path.join(label_path, img_file)
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.resize(img, target_size)
                    images.append(img)
                    labels.append(label)
    return np.array(images), np.array(labels)

# 2. Adjust Lighting Conditions
def adjust_brightness(image, factor):
    """Adjust the brightness of an image by a given factor."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def generate_lighting_conditions(images):
    """Generates normal, dim, and bright versions of the images."""
    normal = images
    dim = np.array([adjust_brightness(img, 0.5) for img in images])  # Reduce brightness
    bright = np.array([adjust_brightness(img, 1.5) for img in images])  # Increase brightness
    return normal, dim, bright

# 3. Prepare Dataset
dataset_path = "path/to/SoF_dataset"
images, labels = load_images(dataset_path)
labels, class_names = np.unique(labels, return_inverse=True)  # Convert labels to integers

normal, dim, bright = generate_lighting_conditions(images)

# Split into training and testing sets (using only normal lighting for training)
X_train, X_test, y_train, y_test = train_test_split(normal, class_names, test_size=0.3, random_state=42)
X_test_dim = dim[:len(X_test)]  # Corresponding dim lighting test set
X_test_bright = bright[:len(X_test)]  # Corresponding bright lighting test set

# Normalize images for the model
X_train, X_test, X_test_dim, X_test_bright = [
    x.astype('float32') / 255.0 for x in [X_train, X_test, X_test_dim, X_test_bright]
]

y_train_cat = to_categorical(y_train, num_classes=len(np.unique(class_names)))

# 4. Build and Train the Model
def build_model(input_shape, num_classes):
    """Builds a simple CNN model for facial recognition."""
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = build_model(X_train.shape[1:], len(np.unique(class_names)))
model.fit(X_train, y_train_cat, epochs=10, batch_size=32, validation_split=0.2)

# 5. Evaluate the Model
def evaluate_model(model, X, y_true, condition_name):
    """Evaluate the model on a test set and print results."""
    y_pred = np.argmax(model.predict(X), axis=1)
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Accuracy under {condition_name} lighting: {accuracy * 100:.2f}%")
    print(classification_report(y_true, y_pred, target_names=np.unique(labels)))

    # Score distribution plot
    y_scores = model.predict(X)
    fpr, tpr, _ = roc_curve(y_true, y_scores[:, 1], pos_label=1)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{condition_name} (AUC = {roc_auc:.2f})')
    return accuracy

plt.figure()
evaluate_model(model, X_test, y_test, "Normal")
evaluate_model(model, X_test_dim, y_test, "Dim")
evaluate_model(model, X_test_bright, y_test, "Bright")
plt.plot([0, 1], [0, 1], linestyle='--', label='Chance')
plt.legend()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Different Lighting Conditions')
plt.show()
