import tensorflow as tf
from tensorflow.keras import layers, models # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.applications import EfficientNetB0 # type: ignore
import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model, save_model
from tensorflow.keras.applications.efficientnet import preprocess_input # type: ignore

def efficientnet_model(input_shape, num_classes):
    base_model = EfficientNetB0(input_shape=input_shape, include_top=False, weights='imagenet')

    base_model.trainable = False

    for layer in base_model.layers[-20:]:
        layer.trainable = True


    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])

    return model

input_shape_efficientnet = (224, 224, 3)

num_classes_efficientnet = 2

model_efficientnet = efficientnet_model(input_shape_efficientnet, num_classes_efficientnet)

model_efficientnet.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model_efficientnet.summary()

# train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
# test_datagen = ImageDataGenerator(rescale=1./255)

datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=30,
    zoom_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

train_generator = datagen.flow_from_directory(
    r"C:\Users\hp\Downloads\archivedataset\train",
    target_size=(224, 224),
    batch_size=32,
    class_mode='sparse'
)
print(f"Number of training batches: {len(train_generator)}")
print(f"Number of training samples: {len(train_generator.classes)}")
print(train_generator.class_indices)


test_generator = datagen.flow_from_directory(
    r"C:\Users\hp\Downloads\archivedataset\test",
    target_size=(224, 224),
    batch_size=32,
    class_mode='sparse'
)
print(f"Number of testing batches: {len(test_generator)}")
print(f"Number of testing samples: {len(test_generator.classes)}")
print(test_generator.class_indices)

# train_generator = train_datagen.flow_from_directory(r"C:\Users\hp\Documents\skin_disease\skindiseaseprediction\Dataset\Train",
#                                                    target_size=input_shape_efficientnet[:2],
#                                                    batch_size=32, class_mode='sparse')


#test_generator = test_datagen.flow_from_directory(r"C:\Users\hp\Documents\skin_disease\skindiseaseprediction\Dataset\Test",
#                                                 target_size=input_shape_efficientnet[:2],
#                                                 batch_size=32, class_mode='sparse')


from sklearn.utils.class_weight import compute_class_weight
import numpy as np

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)

class_weights = dict(enumerate(class_weights))
print(class_weights)


if len(train_generator) == 0 or len(test_generator) == 0:
    print("No data found. Check your directory paths and data.")
else:
    history = model_efficientnet.fit(train_generator, epochs=10, validation_data=test_generator, class_weight=class_weights)

    model_efficientnet.save("skinDisease.keras")

    plt.figure(figsize=(12, 6))
    plt.plot(history.history['accuracy'], label='Training Accuracy', marker='o')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='o')
    plt.title('Training and Validation Accuracy Across Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

    accuracy_efficientnet = model_efficientnet.evaluate(test_generator)[1]
    print(f"EfficientNet Accuracy on Testing Set: {accuracy_efficientnet * 100:.2f}%")
