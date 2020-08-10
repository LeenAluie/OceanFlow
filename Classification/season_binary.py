import numpy as np
import xarray as xr
from tensorflow.python import keras
from tensorflow.python.keras import models, layers
import matplotlib.pyplot as plt

# Load the data. Data was downloaded from CMEMS website in NetCDF format.
# It includes daily surface temperature, longitude, and latitude data over the entire year of 2019.
# Data does not cover the entire earth, but just a specific region that got selected.

filepath = '/Users/Leen1/Desktop/deeplearning/OceanFlow/YearlyData/'
data = xr.open_mfdataset(filepath+'*_surface', combine='by_coords')

depth = data.depth
time = data.time
lons, lats = data.longitude, data.latitude
temp = data.thetao

# Create numpy arrays of the data.

d = depth.values
lat, lon = lats.values, lons.values
T = temp.values
t = time.values

# Delete the redundant 1st day of 5th month. It was downloaded twice by mistake.

t = np.delete(t, np.s_[120], axis=0)
T = np.delete(T, np.s_[120], axis=0)

# Re-shape the input data into a format that could be input into a dense layer.
# The input should be of shape (samples, features). The samples in this case would correspond
# to the time-span of the data (365 days in this case). The features would correspond to the
# latitude x longitude dimensions (data resolution).

T2 = T.reshape(len(t), lat.shape[0]*lon.shape[0])

# Drop all the Nan's from the data. Nan's in this case correspond to land areas, so no need to
# average or interpolate.

T3 = np.zeros(shape=(len(t), T2.shape[1] - sum(np.isnan(T2[0]))), dtype='float32')

for i in range(len(t)):
    T3[i] = T2[i, np.logical_not(np.isnan(T2[i]))]

# Define the x-data to be the temperature data.

xdata = T3

# Prepare the labels/targets of the x-data. They should be binary in this case.
# Winter = 0 and Summer = 1. Consider months of April-September to be summer and the rest winter.

ydata = np.zeros(shape=(len(t),), dtype='float32')

for i in range(len(t)):
    if ( (t[i] >=  np.datetime64('2019-04')) and (t[i] <= np.datetime64('2019-09')) ):
        ydata[i] = 1.

# Data needs to be shuffled before being fed into neural network.
# To shuffle x-data and y-data simultaneously, join them together, shuffle, and then split again.

ydata = ydata[:,np.newaxis]
joined = np.concatenate((ydata, xdata), axis=1)
np.random.shuffle(joined)
ydata2, xdata2 = np.split(joined,[1], axis=1)
ydata2 = ydata2.reshape((len(t),))

# Split the data into training and testing data.

xtrain, ytrain = xdata2[:300], ydata2[:300]
# xval, yval = xdata2[200:300], ydata2[200:300]
xtest, ytest = xdata2[300:], ydata2[300:]

# Normalize the data first to make classification easier.
# Normalization for the test data is done using stats of the train data.

mean = xtrain.mean(axis=0)
xtrain -= mean
# xval -= mean
xtest -= mean

std = xtrain.std(axis=0)
xtrain /= std
# xval /= std
xtest /= std

# Define the network model.

def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu',
                            input_shape=(xtrain.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
    return model

# Preparing validation data using K-fold method, due to sparsity of the training data.

k = 4                       # number of folds to use
num_val_samples = len(xtrain) // k
num_epochs = 15
# all_scores = []           # used to save val mae per val fold
all_acc_histories = []      # used to save val mae of model per epoch

for i in range(k):
    print('processing fold #', i)
    xval = xtrain[i * num_val_samples: (i + 1) * num_val_samples]
    yval = ytrain[i * num_val_samples: (i + 1) * num_val_samples]
    partial_xtrain = np.concatenate( [xtrain[:i * num_val_samples],
                                        xtrain[(i + 1) * num_val_samples:]], axis=0)
    partial_ytrain = np.concatenate( [ytrain[:i * num_val_samples],
                                        ytrain[(i + 1) * num_val_samples:]], axis=0)
    model = build_model()
    history = model.fit(partial_xtrain, partial_ytrain,
                        validation_data=(xval, yval),
                        epochs=num_epochs, batch_size=16, verbose=1)
    # val_loss, val_acc = model.evaluate(xval, yval, verbose=1)
    # all_scores.append(val_acc)
    acc_history = history.history['val_acc']
    all_acc_histories.append(acc_history)

# calculate average accuracy
average_acc_history = [np.mean([x[i] for x in all_acc_histories]) for i in range(num_epochs)]

# smooth out accuracy validation scores and exclude first 10 points, in order to make plot clearer.

def smooth_curve(points, factor=0.9):
  smoothed_points = []
  for point in points:
    if smoothed_points:
      previous = smoothed_points[-1]
      smoothed_points.append(previous * factor + point * (1 - factor))
    else:
      smoothed_points.append(point)
  return smoothed_points

# smooth_acc_history = smooth_curve(average_acc_history[10:])
#
# plt.plot(range(1, len(smooth_acc_history) + 1), smooth_acc_history)
# plt.xlabel('Epochs')
# plt.ylabel('Validation Accuracy')
# plt.show()

# Train the final model after having optimized the model parameters (such as epochs num, num of layers, etc...)
# This time use the entire training data and test on the testing data.

model = build_model()
model.fit(xtrain, ytrain, epochs=20, batch_size=16, verbose=1)
test_loss, test_acc = model.evaluate(xtest, ytest)
preds = model.predict(xtest)
