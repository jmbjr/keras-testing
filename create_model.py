import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import  Activation
from keras.layers.core import Dense
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
import create_samples as cs

trained_model_filename = 'fake_med_trial.h5'

train_samples, train_labels, scaled_train_samples = cs.create_samples(13, 64, 100, 5000, 0.95)

#first layer, only, needs to know input shape of the input data
model = Sequential([Dense(16, input_shape=(1,), activation='relu'),
                    Dense(32, activation='relu'),
                    Dense(2, activation='softmax')])

model.summary()

model.compile(optimizer=Adam(lr=0.01),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x=scaled_train_samples,
          y=train_labels,
          batch_size=100,
          epochs=50,
          shuffle=True,
          verbose=2,
          validation_split=0.1)

print('Saving model to {}'.format(trained_model_filename))
model.save(trained_model_filename)


