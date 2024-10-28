# Transfer-Learning-for-binary-classification
## Aim
To Implement Transfer Learning for Horses_vs_humans dataset classification using InceptionV3 architecture.
## Problem Statement and Dataset
The objective of this project is to classify images from the Horses vs. Humans dataset using the InceptionV3 architecture through transfer learning. This dataset presents a binary classification challenge, distinguishing between images of horses and humans. By leveraging pre-trained weights from the InceptionV3 model, we aim to enhance classification accuracy and reduce training time. The project will evaluate the model's performance based on metrics such as accuracy, precision, and recall. Ultimately, the goal is to demonstrate the effectiveness of transfer learning in image classification tasks.
![image](https://github.com/user-attachments/assets/ee8f1838-891e-4317-8176-41946084867e)
</br>
</br>
</br>

## DESIGN STEPS
### STEP 1:
Load the pre-trained InceptionV3 model, freeze its layers, and prepare the horse and human datasets.

### STEP 2:
Add custom layers for binary classification, then compile the model with the RMSprop optimizer.


### STEP 3:

Train the model using augmented data, apply early stopping, and plot training/validation accuracy and loss.


## PROGRAM
Include your code here
```python
# Import all the necessary files!
import os
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
from os import getcwd

LOCAL_WEIGHTS_FILE = './model/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
pre_trained_model = tf.keras.applications.inception_v3.InceptionV3(
    include_top=False,
    input_shape=(150, 150, 3),
    weights='imagenet'
    )
for layer in pre_trained_model.layers:
  layer.trainable = False
# Print the model summary
# Write Your Code
print('Name: D VERGIN JENIFER          Register Number: 212223240174')

last_desired_layer = pre_trained_model.get_layer('mixed7')
last_output = last_desired_layer.output

print('last layer output shape: ', last_output.shape)
class EarlyStoppingCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs['accuracy']>0.999:
            self.model.stop_training = True
            print("\nReached 99.9% accuracy so cancelling training!")

from tensorflow.keras.optimizers import RMSprop

x = tf.keras.layers.Flatten()(last_output)
x = tf.keras.layers.Dense(1024, activation='relu')(x)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.Dense(1, activation='sigmoid')(x)
model = tf.keras.Model(inputs=pre_trained_model.input, outputs=x)
model.compile(
    optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.00001),
    loss='binary_crossentropy',  # Use binary crossentropy for binary classification
    metrics=['accuracy']
    )
print('Name: D VERGIN JENIFER          Register Number: 212223240174')
# Get the Horse or Human dataset
path_horse_or_human = '/content/horse-or-human.zip'
# Get the Horse or Human Validation dataset
path_validation_horse_or_human = '/content/validation-horse-or-human.zip'
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import zipfile

local_zip = path_horse_or_human
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/tmp/training')
zip_ref.close()

local_zip = path_validation_horse_or_human
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/tmp/validation')
zip_ref.close()
from google.colab import drive
drive.mount('/content/drive')
# Define our example directories and files
train_dir = '/tmp/training'
validation_dir = '/tmp/validation'

train_horses_dir = os.path.join(train_dir, 'horses')
train_humans_dir = os.path.join(train_dir, 'humans')
validation_horses_dir = os.path.join(validation_dir, 'horses')
validation_humans_dir = os.path.join(validation_dir, 'humans')

train_horses_fnames = os.listdir(train_horses_dir)
train_humans_fnames = os.listdir(train_humans_dir)
validation_horses_fnames = os.listdir(validation_horses_dir)
validation_humans_fnames = os.listdir(validation_humans_dir)

print(len(train_horses_fnames))
print(len(train_humans_fnames))
print(len(validation_horses_fnames))
print(len(validation_humans_fnames))
train_datagen = ImageDataGenerator(rescale = 1/255,
                                  height_shift_range = 0.2,
                                  width_shift_range = 0.2,
                                  horizontal_flip = True,
                                  vertical_flip = True,
                                  rotation_range = 0.4,
                                  shear_range = 0.1,
                                  zoom_range = 0.3,
                                  fill_mode = 'nearest'
                                  )

# Note that the validation data should not be augmented!
test_datagen = ImageDataGenerator(rescale = 1/255)

# Flow training images in batches of 20 using train_datagen generator
train_generator = train_datagen.flow_from_directory(train_dir,
                                                   target_size = (150, 150),
                                                   batch_size = 20,
                                                   class_mode = 'binary',
                                                   shuffle = True)

# Flow validation images in batches of 20 using test_datagen generator
validation_generator =  test_datagen.flow_from_directory(validation_dir,
                                                        target_size = (150, 150),
                                                        batch_size =20,
                                                        class_mode = 'binary',
                                                        shuffle = False)
history = model.fit(
    train_generator,
    validation_data = validation_generator,
    epochs = 100,
    verbose = 2,
    callbacks = [EarlyStoppingCallback()],
)
%matplotlib inline
import matplotlib.pyplot as plt
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Name: D Vergin Jenifer           Register Number:212223240174      ')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.figure()
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Name: D Vergin Jenifer           Register Number:212223240174')
plt.title('Training and validation Loss')
plt.legend(loc=0)
plt.figure()


plt.show()
```


## OUTPUT
### Training Accuracy, Validation Accuracy Vs Iteration Plot
Include your plot here
![Screenshot 2024-10-21 052301](https://github.com/user-attachments/assets/55ebb462-7b31-4b7e-bebe-fe467bc23a30)
![image](https://github.com/user-attachments/assets/6feb74d5-c909-4a2b-8d66-51ba89edf533)

![image](https://github.com/user-attachments/assets/4f854c08-2b93-48a4-8bac-56fe890b7b30)

![image](https://github.com/user-attachments/assets/250a5910-7f63-4e75-ac99-3a8a42fc5468)

![image](https://github.com/user-attachments/assets/d9fc600c-ce89-4696-9068-298e6267a055)


### Training Loss, Validation Loss Vs Iteration Plot
![image](https://github.com/user-attachments/assets/9620a494-477b-4363-becd-d019e9d294e6)


### Conclusion
![image](https://github.com/user-attachments/assets/433143a4-6d46-4c0d-bbe9-fb7c930552ef)

![image](https://github.com/user-attachments/assets/c3d215bc-0641-4b1a-9ca5-b9c9433db7aa)


![image](https://github.com/user-attachments/assets/b5cd8851-a1b3-4587-802d-951f65cadce4)

## RESULT
Thus, Implementation of Transfer Learning for Horses_vs_humans dataset classification using InceptionV3 architecture, was successful.
