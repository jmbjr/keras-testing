import numpy as np
import math
from random import randint
from sklearn.preprocessing import MinMaxScaler

def create_samples(min, mid, max, total, pctsplit):
    train_labels = []
    train_samples = []
    half_total = math.floor(total/2)
    half_split = half_total - math.floor(half_total * pctsplit)

    for ii in range(half_total):
        random_younger = randint(min,mid)
        train_samples.append(random_younger)
        train_labels.append(0)

        random_older = randint(mid+1,max)
        train_samples.append(random_older)
        train_labels.append(1)

    for ii in range(half_split):
        random_younger = randint(min,mid)
        train_samples.append(random_younger)
        train_labels.append(1)

        random_older = randint(mid+1,max)
        train_samples.append(random_older)
        train_labels.append(0)

    #print
    for ts,tl in zip(train_samples, train_labels):
        print('{}/{}'.format(ts,tl))

    train_samples = np.array(train_samples)
    train_labels = np.array(train_labels)

    #scale data to between 0 to 1
    scaler = MinMaxScaler(feature_range=(0,1))
    #need to reshape vector data to a -1
    scaled_train_samples = scaler.fit_transform((train_samples).reshape(-1,1))

    randomize = np.arange(len(train_samples))
    np.random.shuffle(randomize)

    train_samples = train_samples[randomize]
    train_labels = train_labels[randomize]
    scaled_train_samples = scaled_train_samples[randomize]

    #print scaled
    for ts,tl,sts in zip(train_samples, train_labels, scaled_train_samples):
        print('{}->{}/{}'.format(ts,sts,tl))

    return train_samples, train_labels, scaled_train_samples

