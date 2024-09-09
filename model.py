import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import os
import pandas as pd
from PIL import Image  


#  target size for resizing
TARGET_SIZE = (64, 64)

# Function to resize and normalize an image, while handling RGBA (4 channels)
def preprocess_image(img_path):
    """
    Resizes an image to TARGET_SIZE and normalizes pixel values to [0, 1].
    Converts RGBA images (4 channels) to RGB (3 channels).
    """
    # Open the image
    img = Image.open(img_path)
    
    # Convert RGBA to RGB 
    if img.mode == 'RGBA':
        img = img.convert('RGB')
    
    # Resize the image to the target size
    img_resized = img.resize(TARGET_SIZE)
    
    # Convert image to numpy array and normalize pixel values (divide by 255)
    img_array = np.array(img_resized) / 255.0
    
    return img_array

# Preprocess the Train Dataset
# Path to the 'Train' folder and Train.csv file
train_folder = r'C:\Users\Hp\OneDrive\Documents\PersonalProjects\RoadsignDetection\Train\\'
train_csv_path = r'C:\Users\Hp\OneDrive\Documents\PersonalProjects\RoadsignDetection\Train.csv'

# Load the Train.csv file to get the image paths and class labels
train_data = pd.read_csv(train_csv_path)

# Preprocess the train dataset
train_images = []
train_labels = []

# Loop through each row in the Train.csv to preprocess each image
for index, row in train_data.iterrows():
    img_path = os.path.join(row['Path'])  
    class_id = row['ClassId']  # Get the class label (ClassId)
    
    # Preprocess the image (resize and normalize)
    img_array = preprocess_image(img_path)
    
    # Append the preprocessed image and label
    train_images.append(img_array)
    train_labels.append(class_id)

# Convert the lists to NumPy arrays
X_train = np.array(train_images)
y_train = np.array(train_labels)

# One-hot encode the train labels (43 classes)
y_train_onehot = to_categorical(y_train, num_classes=43)


# Preprocess the Test Dataset
# Path to the 'Test' folder and Test.csv file
test_folder = r'C:\Users\Hp\OneDrive\Documents\PersonalProjects\RoadsignDetection\Test\\'
test_csv_path = r'C:\Users\Hp\OneDrive\Documents\PersonalProjects\RoadsignDetection\Test.csv'

# Load the Test.csv file to get the image paths and class labels
test_data = pd.read_csv(test_csv_path)

# Preprocess the test dataset
test_images = []
test_labels = []

# Loop through each row in the Test.csv to preprocess each image
for index, row in test_data.iterrows():
    img_path = os.path.join(row['Path'])  
    class_id = row['ClassId']  # Get the class label (ClassId)
    
    # Preprocess the image (resize and normalize)
    img_array = preprocess_image(img_path)
    
    # Append the preprocessed image and label
    test_images.append(img_array)
    test_labels.append(class_id)

# Convert the lists to NumPy arrays
X_test = np.array(test_images)
y_test = np.array(test_labels)

# One-hot encode the test labels (43 classes)
y_test_onehot = to_categorical(y_test, num_classes=43)


# Model Architecture
model = Sequential()

# First Convolutional Layer
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Second Convolutional Layer
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Third Convolutional Layer
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten the feature maps before fully connected layers
model.add(Flatten())

# Dense layer with dropout to prevent overfitting
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))  # Dropout to prevent overfitting

# Output layer with softmax activation for multi-class classification (43 classes)
model.add(Dense(43, activation='softmax'))

# Compile the model
model.compile(optimizer=Adam(), 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# Summary of the model architecture
model.summary()

# Early stopping callback to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model
history = model.fit(
    X_train, y_train_onehot,
    validation_split=0.2,  # Use 20% of the training data for validation
    epochs=10,  
    batch_size=64,  
    callbacks=[early_stopping]  # Early stopping to prevent overfitting
)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(X_test, y_test_onehot)

print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test Loss: {test_loss:.4f}")

# Plotting the training and validation accuracy and loss
import matplotlib.pyplot as plt

# Plot training & validation accuracy values
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='best')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='best')

plt.show()

# Generate a classification report
from sklearn.metrics import classification_report, confusion_matrix

# Predict on the test set
y_pred = model.predict(X_test)

# Convert one-hot encoded labels back to class labels
y_test_labels = np.argmax(y_test_onehot, axis=1)
y_pred_labels = np.argmax(y_pred, axis=1)

# Print classification report
print(classification_report(y_test_labels, y_pred_labels))

# Plot confusion matrix
confusion_mtx = confusion_matrix(y_test_labels, y_pred_labels)

plt.figure(figsize=(10, 8))
sns.heatmap(confusion_mtx, annot=True, fmt='d', cmap='Blues', xticklabels=range(43), yticklabels=range(43))
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()