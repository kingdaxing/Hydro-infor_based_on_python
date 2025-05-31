## This Python script decsribes time series prediction using LSTM
# Prepared by Cai Hejiang, Research fellow, Department of Civil and Environmental Engineering, March 2024.
import os, logging, pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import pip
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers, regularizers
from livelossplot import PlotLossesKeras

krevents = pd.read_csv(r'D:\PXX-NUS\05-S2 NUS CE in SCR\NEWS\04-CE5310-Hydroinformatics\HW5_ANN\Selected_Events.csv')
# Specify the sheet name or index (0-indexed), Index 1 corresponds to the second sheet
# plot events: unit: Q [L/s]; Rainfall[mm] 
rainfall = krevents.iloc[:, 1]
Q01 = krevents.iloc[:, 2]
Q02 = krevents.iloc[:, 3]
Q04 = krevents.iloc[:, 4]
Qcenter = krevents.iloc[:, 5]
Qopp = krevents.iloc[:, 6]
dataset = pd.DataFrame({'rainfall': rainfall, 'Q01': Q01, 'Q02': Q02, 'Q04': Q04,
                       'Qcenter': Qcenter, 'Qopp': Qopp})
dataset

# plot and view as one continue event
mpl.rcParams.update({'font.size': 14})  # font and size setting
fig, [ax1, ax2] = plt.subplots(nrows=2, ncols=1, figsize=(10, 8))
# subplot1 - Q01, Q02, Q04, Qcenter, Qopp
ax1.plot(dataset['Q01'], color='pink', label='Q01')
ax1.plot(dataset['Q02'], color='blue', label='Q02')
ax1.plot(dataset['Q04'], color='green', label='Q04')
# ax1.plot(dataset['Qcenter'], color='black', label='Qcenter')
ax1.plot(dataset['Qopp'], color='orange', label='Qopp')
ax1.set_title("Discharge and Rainfall")
ax1.set_ylabel("Discharge (l/s)")
ax1.legend()

# subplot2 - rainfall
ax2.plot(dataset['rainfall'], color='red')
ax2.set_ylabel("Rainfall (mm)")
ax2.set_xlabel("Index (per min)")
plt.tight_layout()
# plt.savefig(r'D:\PXX-NUS\05-S2 NUS CE in SCR\NEWS\04-CE5310-Hydroinformatics\HW5_ANN\event_plot.png', dpi=300) 
# plt.show()


# define a function to reshape the dataset to meet LSTM model
def get_wrapped_data(dataset, wrap_length=60):
    """
    Wrap the data for the shape requirement of LSTM.

    Parameters
    ----------
    dataset: the pandas dataframe obtained from the function.
    wrap_length: the number of time steps to be considered for the LSTM layer.

    Returns
    ----------
    data_x: the input array where each element is the corresponding wrapped input matrix of each sample.
    data_y: the output array where each element is the corresponding target of each sample.
    """
    data_x, data_y = [], []

    for i in range(len(dataset) - wrap_length):
        # dataset (rainfall, Q01, Q02, Q04, Qopp)
        data_x.append(dataset.iloc[i:i+wrap_length, [0, 1, 2, 4]].to_numpy(dtype='float64'))
        data_y.append(dataset.iloc[i+wrap_length, [3]].astype('float64'))

    return np.array(data_x), np.array(data_y)

# Wrap the data
data_x, data_y = get_wrapped_data(dataset, wrap_length=60)
data_x.shape
data_y.shape
split_index = int(len(data_x) * 0.8) # splitting = 80% test + 20% train

# Split the data into training and testing sets
x_train = data_x[:split_index]
y_train = data_y[:split_index]
x_test = data_x[split_index:]
y_test = data_y[split_index:]

y_train_subset = y_train[:600]
plt.plot( y_train_subset, label='Training Data')
# plt.show()


# Initialize scale parameters
scale_params = {
    "train_x_mean": 0,
    "train_x_std": 1,
    "train_y_mean": 0,
    "train_y_std": 1
}

# Calculate mean and standard deviation for scaling
scale_params["train_x_mean"] = np.mean(x_train, axis=(0, 1))
scale_params["train_x_std"] = np.std(x_train, axis=(0, 1))
scale_params["train_y_mean"] = np.mean(y_train, axis=0)
scale_params["train_y_std"] = np.std(y_train, axis=0)


# Normalize training and testing data
x_train_scaled = (x_train - scale_params["train_x_mean"][None, None, :]) / scale_params["train_x_std"][None, None, :]
y_train_scaled = (y_train - scale_params["train_y_mean"][None, :]) / scale_params["train_y_std"][None, :]
x_test_scaled = (x_test - scale_params["train_x_mean"][None, None, :]) / scale_params["train_x_std"][None, None, :]
y_test_scaled = (y_test - scale_params["train_y_mean"][None, :]) / scale_params["train_y_std"][None, :]

print(f'The shape of x_train, y_train after wrapping by 10-min are {x_train_scaled.shape}, {y_train_scaled.shape}')
print(f'The shape of x_test, y_test after wrapping by 10-min are   {x_test_scaled.shape}, {y_test_scaled.shape}')


# Build up a LSTM model
inputs = layers.Input(x_train.shape[1:], name='input')
lstm   = layers.LSTM(units=5, name='lstm')(inputs)
output = layers.Dense(units=1, name='dense', activation='linear')(lstm)

model  = models.Model(inputs, output)
model.summary()

es     = callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=40, 
                                 min_delta=0.01, restore_best_weights=True)
model.compile(loss='mse', metrics=['mean_absolute_error'], optimizer='adam')
history = model.fit(x_train_scaled, y_train_scaled, epochs=80, validation_split=0.2,
                    callbacks=[es, PlotLossesKeras()])

# Visualize loss and error by 2 rows
from tensorflow.keras.callbacks import Callback
import matplotlib.pyplot as plt

class DynamicPlotCallback(Callback):
    def __init__(self):
        super().__init__()
        self.fig, self.axes = plt.subplots(2, 1, figsize=(10, 6))
        plt.ion()  # 打开交互模式

    def on_epoch_end(self, epoch, logs=None):
        # 更新图像
        epochs = range(epoch + 1)

        # Loss
        self.axes[0].clear()
        self.axes[0].plot(epochs, self.model.history.history["loss"], "orange", label="Training")
        self.axes[0].plot(epochs, self.model.history.history["val_loss"], "c", label="Validation")
        self.axes[0].set_ylabel("Loss")
        self.axes[0].set_title("Loss Over Epochs")
        self.axes[0].legend(loc='upper right')

        # Error
        self.axes[1].clear()
        self.axes[1].plot(epochs, self.model.history.history["mean_absolute_error"], "orange", label="Training abs_Error")
        self.axes[1].plot(epochs, self.model.history.history["val_mean_absolute_error"], "c", label="Validation abs_Error")
        self.axes[1].set_xlabel("Epochs")
        self.axes[1].set_ylabel("abs_Error")
        self.axes[1].legend(loc='upper right')

        plt.tight_layout()
        plt.pause(0.01)

    def on_train_end(self, logs=None):
        plt.ioff()  # 关闭交互模式
        plt.show()

# 使用回调
dynamic_plot = DynamicPlotCallback()
history = model.fit(
    x_train_scaled,
    y_train_scaled,
    epochs=80,
    validation_split=0.2,
    callbacks=[es, dynamic_plot],
    verbose=0
)




# plt.show()
#plt.savefig(r'D:\PXX-NUS\05-S2 NUS CE in SCR\NEWS\04-CE5310-Hydroinformatics\HW5_ANN\loss_1.png', dpi=300) 
#model.save(r'D:\PXX-NUS\05-S2 NUS CE in SCR\NEWS\04-CE5310-Hydroinformatics\HW5_ANN\lstm6_unit_5_wrap60.h5')


# Prediction
pred_train = model.predict(x_train_scaled)
pred_test  = model.predict(x_test_scaled)

Training_result = pred_train * scale_params["train_y_std"] + scale_params["train_y_mean"]
testing_result = pred_test* scale_params["train_y_std"] + scale_params["train_y_mean"]

dataset['flow_pred'] = None
dataset.iloc[60:len(Training_result)+60, dataset.columns.get_loc('flow_pred')] = Training_result
dataset.iloc[len(Training_result)+60:, dataset.columns.get_loc('flow_pred')] = testing_result


plt.figure(figsize=(4, 3)) 
plt.plot(dataset['Q04'], label='Q04_Actual', color='blue')
plt.plot(dataset['flow_pred'], label='Q04_predicted', color='red')
plt.legend()
mpl.rcParams.update({'font.size': 8})  
plt.xlabel('Index (per 60-min wrapped)')  
plt.ylabel('Q_MD04 (l/s)')
plt.title('Q04_Actual vs. Q04_predicted')
plt.tight_layout()
# plt.show()
# dataset.to_csv(r'D:\PXX-NUS\05-S2 NUS CE in SCR\NEWS\04-CE5310-Hydroinformatics\HW5_ANN\lstm6_wrap60.csv')
#plt.savefig(r'D:\PXX-NUS\05-S2 NUS CE in SCR\NEWS\04-CE5310-Hydroinformatics\HW5_ANN\pred_10_wrapped.png', dpi=300) 


# assessment and accuracy
actual_data = dataset['Q04'].values[len(Training_result)+60:]
denormalized_predictions = dataset['flow_pred'].values[len(Training_result)+60:]
# accuracy
deviation = (actual_data - denormalized_predictions) / actual_data
accuracy_rate = 1 - np.abs(np.mean(deviation))
accuracy_rate

# RMSE
RMSE = np.sqrt(np.mean((actual_data - denormalized_predictions) ** 2))
RMSE

