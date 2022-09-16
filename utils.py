from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import pandas as pd
from tqdm import tqdm
import numpy as np


def scale_data(df, column):
    scaled_df = pd.DataFrame()
    for customer in tqdm(df["AP"].unique()):
        scaler = MinMaxScaler()
        customer_df = df[df["AP"]==customer]
        customer_df["scaled"] = scaler.fit_transform(customer_df[[column]])
        scaled_df = pd.concat((scaled_df, customer_df), axis=0).sort_index()
    #scaled_df.drop(["ENERGY"], axis=1, inplace=True)

    return scaled_df

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def generate_lagged_features(df, var, lagged_features):
        
    for t in range(1, lagged_features+1):
        df['lag'+str(t)] = df[var].shift(t)


def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred)/y_true)*100)


def smape_adjusted(y_true, y_pred):
    return (1/y_true.size * np.sum(np.abs(y_pred-y_true) / (np.abs(y_true) + np.abs(y_pred))*100))

def gmrae(y_true, y_pred, y_naive):

    abs_scaled_errors = np.abs(y_true - y_pred)/np.abs(y_true - y_naive)
    return np.exp(np.mean(np.log(abs_scaled_errors)))

def nrmse(y_true, y_pred, naive):
    rmse_model = mean_squared_error(y_true=y_true, y_pred=y_pred, squared=False)
    rmse_naive = mean_squared_error(y_true=y_true, y_pred=naive, squared=False)

    return rmse_model/rmse_naive