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
from keras.utils import plot_model

#dumb copy/paste of image prep stuff
train_path = 'cnn/things/train'
valid_path = 'cnn/things/valid'
test_path = 'cnn/things/test'

thing_classes = ['pigeon', 'rooster', 'dragonfly', 'mayfly', 'ibis']
num_things = len(thing_classes)

train_basesize = 20
valid_basesize = 8
test_basesize = 5

train_dataset_size = num_things * train_basesize
train_batch_size = 2 #10
train_epoch_steps = train_dataset_size/train_batch_size

valid_dataset_size = num_things * valid_basesize
valid_batch_size = 2 #4
valid_epoch_steps = valid_dataset_size/valid_batch_size

test_dataset_size = num_things * test_basesize
test_batch_size = test_dataset_size
test_epoch_steps = test_dataset_size/test_batch_size

target_size = (224,224)

train_batches = ImageDataGenerator().flow_from_directory(train_path,
                                                         target_size=target_size,
                                                         classes=thing_classes,
                                                         batch_size=train_batch_size)
valid_batches = ImageDataGenerator().flow_from_directory(valid_path,
                                                         target_size=target_size,
                                                         classes=thing_classes,
                                                         batch_size=valid_batch_size)
test_batches = ImageDataGenerator().flow_from_directory(test_path,
                                                         target_size=target_size,
                                                         classes=thing_classes,
                                                         batch_size=test_batch_size)


#plt.show()



#pre-trained model
vgg16_model = keras.applications.vgg16.VGG16()
vgg16_model.summary()
print(type(vgg16_model))
#plot_model(vgg16_model, to_file='vgg16.png',  show_shapes=True)

#pop last layer since we only want 2 outputs
vgg16_model.layers.pop()

#copy vgg16 to new model
model = Sequential()
for layer in vgg16_model.layers:
    model.add(layer)

model.summary()

#recover some memory
del vgg16_model

#exlude layers from further training
for layer in model.layers:
    layer.trainable = False

#modify model to only classify 2 things
model.add(Dense(num_things, activation='softmax'))

model.summary()
#note, can't plot input layer for Sequential models
#see https://github.com/keras-team/keras/issues/10638
#it works for vgg16 model since it's not Sequential
#plot_model(model, to_file='vgg16-2outputs.png', show_shapes=True)

model.compile(optimizer=Adam(lr=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#train
imgs, labels = next(train_batches)
#sklearn_util.plots(imgs, titles=labels)


model.fit_generator(train_batches,
                    steps_per_epoch=train_epoch_steps,
                    validation_data=valid_batches,
                    validation_steps=valid_epoch_steps,
                    epochs=100,
                    verbose=2)


#predict
test_imgs, test_labels = next(test_batches)
#sklearn_util.plots(test_imgs, titles=test_labels)

predictions = model.predict_generator(test_batches,
                                      steps=test_epoch_steps,
                                      verbose=2)
rounded_pred = np.round(keras.utils.normalize(predictions))

print(predictions)

#need this so the cm doesn't plot on the above
#probably could add this to the plot_confusion_matrix function?
cm_plot = plt.figure()
cm_plot.show()

#decode one-hot to get final list
test_decode_labels = [np.where(r==1)[0][0] for r in test_labels]
pred_decode_labels = [np.where(r==1)[0][0] for r in rounded_pred]

cm = confusion_matrix(test_decode_labels, pred_decode_labels)
#note, we need to reverse the classes because... well, dunno, but it's not right
sklearn_util.plot_confusion_matrix(cm, thing_classes[::-1], title='Confusion Matrix')
plt.show()

