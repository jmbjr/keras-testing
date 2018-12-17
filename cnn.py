import numpy as np
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense, Flatten
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import *
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
import sklearn_util

train_path = 'cnn/birds/train'
valid_path = 'cnn/birds/valid'
test_path = 'cnn/birds/test'

train_dataset_size = 40
train_batch_size = 10
train_epoch_steps = train_dataset_size/train_batch_size

valid_dataset_size = 16
valid_batch_size = 4
valid_epoch_steps = valid_dataset_size/valid_batch_size

test_dataset_size = 10
test_batch_size = 10
test_epoch_steps = test_dataset_size/test_batch_size


target_size = (250,250)
bird_classes = ['pigeon', 'rooster']
train_batches = ImageDataGenerator().flow_from_directory(train_path,
                                                         target_size=target_size,
                                                         classes=bird_classes,
                                                         batch_size=train_batch_size)
valid_batches = ImageDataGenerator().flow_from_directory(valid_path,
                                                         target_size=target_size,
                                                         classes=bird_classes,
                                                         batch_size=valid_batch_size)
test_batches = ImageDataGenerator().flow_from_directory(test_path,
                                                         target_size=target_size,
                                                         classes=bird_classes,
                                                         batch_size=test_batch_size)

imgs, labels = next(train_batches)
sklearn_util.plots(imgs, titles=labels)
#plt.show()


#build model
rgb = (3,)
model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=target_size + rgb),
        Flatten(),
        Dense(2, activation='softmax')])

model.compile(optimizer=Adam(lr=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])



model.fit_generator(train_batches,
                    steps_per_epoch=train_epoch_steps,
                    validation_data=valid_batches,
                    validation_steps=valid_epoch_steps,
                    epochs=5,
                    verbose=2)

#predict
test_imgs, test_labels = next(test_batches)
sklearn_util.plots(test_imgs, titles=test_labels)


test_labels = test_labels[:,0]
print(test_labels)

predictions = model.predict_generator(test_batches,
                                      steps=test_epoch_steps,
                                      verbose=2)

print(predictions)

#need this so the cm doesn't plot on the above
#probably could add this to the plot_confusion_matrix function?
cm_plot = plt.figure()
cm_plot.show()

cm = confusion_matrix(test_labels, predictions[:,0])
#note, we need to reverse the classes because... well, dunno, but it's not right
sklearn_util.plot_confusion_matrix(cm, bird_classes[::-1], title='Confusion Matrix')
plt.show()

