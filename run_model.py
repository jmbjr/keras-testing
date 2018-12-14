from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import create_samples as cs
import sklearn_util
from keras.models import load_model

trained_model_filename = 'fake_med_trial.h5'
print('Loading model from {}'.format(trained_model_filename))
model = load_model(trained_model_filename)
model.summary()

test_samples, test_labels, scaled_test_samples = cs.create_samples(1, 64, 120, 500, 0.95)

predictions = model.predict(x=scaled_test_samples,
                            batch_size=10,
                            verbose=2)

rounded_predictions = model.predict_classes(x=scaled_test_samples,
                            batch_size=10,
                            verbose=2)

for pp,rp in zip(predictions,rounded_predictions):
    print('{}->{}'.format(pp, rp))

cm = confusion_matrix(y_true=test_labels,
                      y_pred=rounded_predictions)

cm_plot_labels = ['no_side_effects', 'had_side_effects']
sklearn_util.plot_confusion_matrix(cm, cm_plot_labels, title='Confusion Matrix')
plt.show()
