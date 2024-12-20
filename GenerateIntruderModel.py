from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dropout, AveragePooling2D, Conv2D, Flatten, Dense, Input, BatchNormalization, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
import matplotlib.pyplot as plt
import numpy as np
import os
import random

#--------------Initialize Variables-------------

data = []
labels = []

rawDataset = "./dataset"
classifications = ["Intruder", "Permitted"]

#---------Load and Process Dataset---------------

for category in classifications:
    path = os.path.join(rawDataset, category)
    for img in os.listdir(path):
        # Skip .DS_Store files as they are not supported
        if img.startswith('.'):
            continue
        img_path = os.path.join(path, img)
        image = load_img(img_path, target_size=(224, 224))
        image = img_to_array(image)
        image = preprocess_input(image)
        data.append(image)
        labels.append(classifications.index(category))

data = np.array(data, dtype="float32")
labels = np.array(labels)

(trainX, testX, trainY, testY) = train_test_split(data, labels,
    test_size=0.20, stratify=labels, random_state=42)

# Augment images to diversify dataset
augmentedDataset = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.2,
    shear_range=0.2,
    brightness_range=(0.8, 1.2),
    channel_shift_range=20.0,
)

# Visualize augmented images and their corresponding labels
augmentedImages, augmentedLabels = next(augmentedDataset.flow(trainX, trainY, batch_size=9))

# Show sample of augmentedImages and augmentedLabels
plt.figure(figsize=(10, 10))
for i in range(9):
    # Generate a random index within the batch size
    ranInt = random.randint(0, len(augmentedImages) - 1)
    plt.subplot(3, 3, i + 1)
    plt.imshow(np.clip(augmentedImages[ranInt], 0, 1))
    # Convert label to class name
    label_index = int(augmentedLabels[ranInt]) if len(augmentedLabels[ranInt].shape) == 0 else np.argmax(augmentedLabels[ranInt])
    label = classifications[label_index]
    plt.title(label)
    plt.axis("off")
plt.suptitle("Sample Images after Data Augmentation")
plt.show()


#---------Setup ResNet50 Deep Learning Model---------------

# Load the ResNet50 network without top fully connected layers
deepLearningModel = ResNet50(weights="imagenet", include_top=False,
                             input_tensor=Input(shape=(224, 224, 3)))

# Construct top layers of the model with regularization to help prevent overfitting
topLayers = deepLearningModel.output
topLayers = Conv2D(256, (3, 3), padding="same", kernel_regularizer=l2(0.01))(topLayers)
topLayers = BatchNormalization()(topLayers)
topLayers = Activation("relu")(topLayers)
topLayers = AveragePooling2D(pool_size=(2, 2))(topLayers)
topLayers = Flatten()(topLayers)
topLayers = Dense(128, activation="relu", kernel_regularizer=l2(0.01))(topLayers)
topLayers = Dropout(0.5)(topLayers)
topLayers = Dense(1, activation="sigmoid", kernel_regularizer=l2(0.01))(topLayers)

# Freeze the base layers of ResNet50
for layer in deepLearningModel.layers:
    layer.trainable = False

# Combine the base model and the top layers
model = Model(inputs=deepLearningModel.input, outputs=topLayers)

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Define callbacks
early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

def scheduler(epoch, lr):
    return lr * 0.1 if epoch > 10 else lr

lr_scheduler = LearningRateScheduler(scheduler)

#---------Train Model with 3-fold Stratified Cross-Validation---------------

skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
fold_no = 1
histories = []

for train_index, val_index in skf.split(trainX, trainY):
    x_train_fold, x_val_fold = trainX[train_index], trainX[val_index]
    y_train_fold, y_val_fold = trainY[train_index], trainY[val_index]

    # Train model
    train_generator = augmentedDataset.flow(x_train_fold, y_train_fold, batch_size=64)
    history = model.fit(
        train_generator,
        steps_per_epoch=len(x_train_fold) // 64,
        validation_data=(x_val_fold, y_val_fold),
        epochs=30,
        verbose=1,
        callbacks=[early_stopping, lr_scheduler]
    )

    histories.append(history)
    fold_no += 1

#---------Evaluate Model on Test Set---------------

predProbs = model.predict(testX, batch_size=64).flatten()
predIdxs = (predProbs > 0.5).astype("int32")  

# Calculate Test Accuracy
test_accuracy = np.mean(predIdxs == testY)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Show classification report
print(classification_report(testY, predIdxs, target_names=classifications))

# Plot confusion matrix
conf_matrix = confusion_matrix(testY, predIdxs)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=classifications)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.show()

    #---------Save Model---------------
model.save("intruder_detector_resnet.keras")

# Calculate ROC curve and AUC for binary classification
fpr, tpr, _ = roc_curve(testY, predProbs)
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], "k--") 
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.savefig("roc_curve.png")
plt.show()

# plot the training loss and accuracy
N = min(len(history.history["loss"]),
        len(history.history["val_loss"]),
        len(history.history["accuracy"]),
        len(history.history["val_accuracy"]))
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), history.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), history.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), history.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), history.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot_resnet.png")
plt.show()
