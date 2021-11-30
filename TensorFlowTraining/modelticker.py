import logging
from dotenv import load_dotenv
load_dotenv()
import azure.functions as func
import os
import json
import time
from datetime import datetime
import pytz
EST=pytz.timezone('US/Eastern')
import pandas as pd
import numpy as np
import random
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from collections import deque
import pyodbc
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import ModelCheckpoint
from azure.storage.blob import BlobClient
from sklearn.metrics import accuracy_score
try:
    import tempfile
except:
    pass


"""
Modified source code from tutorial https://www.thepythoncode.com/article/stock-price-prediction-in-python-using-tensorflow-2-and-keras to create, train, test a model  using Tensorflow and Keras
"""
class buildmodel():
    def __init__(self,tkr=None,data_df=None):
        self.model_name = f"{tkr}-{LOSS}-{OPTIMIZER}-{CELL.__name__}-seq-{N_STEPS}-step-{LOOKUP_STEP}-layers-{N_LAYERS}-units-{UNITS}"
        self.data_df=data_df

    def load_data(self, n_steps=50, scale=True, shuffle=True,test_size=0.2):
        """
        Imports data from CSV file created in the intradayanalysis.py file and stored in blob storage. 
        """
        lookup_step=LOOKUP_STEP
               
        futureval=(self.data_df['Close']-self.data_df['Close'].shift(lookup_step))/self.data_df['Close'].shift(lookup_step)
        self.data_df.loc[lookup_step:,'FutureValue']=futureval
        
        #Add columns to the feature columns. Default is a minimal set of OHLC only. All columns in the dataframe will be added to this list
        global FEATURE_COLUMNS
        for col in self.data_df.columns:
            if col not in FEATURE_COLUMNS:
                FEATURE_COLUMNS.append(col)
        df=self.data_df.copy()
        df.dropna(inplace=True,axis=0)
        # this will contain all the elements we want to return from this function
        result = {}

        result['df'] = df.copy()

        for col in FEATURE_COLUMNS:
            assert col in df.columns, f"'{col}' does not exist in the dataframe."

        if scale:
            column_scaler = {}
            # scale the data (prices) from 0 to 1 for neural network
            for column in FEATURE_COLUMNS:
                scaler = preprocessing.MinMaxScaler()
                df[column] = scaler.fit_transform(np.expand_dims(df[column].values, axis=1))
                column_scaler[column] = scaler

            # add the MinMaxScaler instances to the result returned
            result["column_scaler"] = column_scaler
        
        # add the target column (label) by shifting by `lookup_step`
        df['future'] = df['FutureValue'].shift(-lookup_step)

        # last `lookup_step` columns contains NaN in future column
        # get them before droping NaNs
        last_sequence = np.array(df[FEATURE_COLUMNS].tail(lookup_step))
        
        # drop NaNs
        df.dropna(inplace=True)

        sequence_data = []
        sequences = deque(maxlen=n_steps)

        for entry, target in zip(df[FEATURE_COLUMNS].values, df['future'].values):
            sequences.append(entry)
            if len(sequences) == n_steps:
                sequence_data.append([np.array(sequences), target])

        # get the last sequence by appending the last `n_step` sequence with `lookup_step` sequence
        # for instance, if n_steps=50 and lookup_step=10, last_sequence should be of 60 (that is 50+10) length
        # this last_sequence will be used to predict future stock prices not available in the dataset
        last_sequence = list(sequences) + list(last_sequence)
        last_sequence = np.array(last_sequence)
        # add to result
        result['last_sequence'] = last_sequence
        
        # construct the X's and y's
        X, y = [], []
        for seq, target in sequence_data:
            X.append(seq)
            y.append(target)

        # convert to numpy arrays
        X = np.array(X)
        y = np.array(y)

        # reshape X to fit the neural network
        X = X.reshape((X.shape[0], X.shape[2], X.shape[1]))
        
        # split the dataset
        result["X_train"], result["X_test"], result["y_train"], result["y_test"] = train_test_split(X, y, 
                                                                                    test_size=test_size, shuffle=shuffle)
        # return the result
        return result

    def create_model(self,sequence_length, units=256, cell=LSTM, n_layers=2, dropout=0.3,
                loss="mean_absolute_error", optimizer="rmsprop", bidirectional=False):
        model = Sequential()
        for i in range(n_layers):
            if i == 0:
                # first layer
                if bidirectional:
                    model.add(Bidirectional(cell(units, return_sequences=True), input_shape=(None, sequence_length)))
                else:
                    model.add(cell(units, return_sequences=True, input_shape=(None, sequence_length)))
            elif i == n_layers - 1:
                # last layer
                if bidirectional:
                    model.add(Bidirectional(cell(units, return_sequences=False)))
                else:
                    model.add(cell(units, return_sequences=False))
            else:
                # hidden layers
                if bidirectional:
                    model.add(Bidirectional(cell(units, return_sequences=True)))
                else:
                    model.add(cell(units, return_sequences=True))
            # add dropout after each layer
            model.add(Dropout(dropout))
        model.add(Dense(1, activation="linear"))
        model.compile(loss=loss, metrics=["mean_absolute_error"], optimizer=optimizer)
        return model

    def train_data(self):

        blob = BlobClient.from_connection_string(conn_str=os.environ.get('blob_conn_str'), container_name="tensorflow", blob_name="results"+"/"+self.model_name+".h5")
        with tempfile.TemporaryDirectory() as tmpdirname:
            with open(tmpdirname + "/" + self.model_name, "wb") as modeldata:

                data = self.load_data(n_steps=N_STEPS, test_size=TEST_SIZE,
                        shuffle=False)

                # construct the model
                model = self.create_model(N_STEPS, loss=LOSS, units=UNITS, cell=CELL, n_layers=N_LAYERS,
                                    dropout=DROPOUT, optimizer=OPTIMIZER, bidirectional=BIDIRECTIONAL)
                
                # some tensorflow callbacks
                checkpointer = ModelCheckpoint(tmpdirname+"/"+self.model_name, save_weights_only=True, save_best_only=True, verbose=1)
                # tensorboard = TensorBoard(log_dir="/logs"+"/"+self.model_name)
                history = model.fit(data["X_train"], data["y_train"],
                                    batch_size=BATCH_SIZE,
                                    epochs=EPOCHS,
                                    validation_data=(data["X_test"], data["y_test"]),
                                    callbacks=[checkpointer],
                                    verbose=1)
                    
                model.save(tmpdirname+"/"+ self.model_name + ".h5")
                
                with open(tmpdirname + "/" + self.model_name + ".h5", "rb") as modeldatabytes:
                    if blob.exists():
                        blob.delete_blob()
                    blob.upload_blob(modeldatabytes)

        trainingparams = "{" + f""""N_STEPS" : {N_STEPS} 
                    , "LOOKUP_STEP" : {LOOKUP_STEP}
                    , "TEST_SIZE" : {TEST_SIZE}
                    , "N_LAYERS" : {N_LAYERS}
                    , "UNITS" : {UNITS}
                    , "DROPOUT" : {DROPOUT}
                    , "BIDIRECTIONAL" : "False"
                    , "LOSS" : "{LOSS}"
                    , "OPTIMIZER" : "{OPTIMIZER}"
                    , "BATCH_SIZE" : {BATCH_SIZE}
                    , "EPOCHS" : {EPOCHS}""" + "}"

        conn_str='DRIVER={ODBC Driver 17 for SQL Server};SERVER='+os.environ.get('server')+ \
            ';DATABASE='+os.environ.get('database')+ \
                ';UID='+os.environ.get('dbusername')+ \
                    ';PWD='+ os.environ.get('dbpassword')
        self.cnxn = pyodbc.connect(conn_str)       
        self.cursor = self.cnxn.cursor()
        currentdate=datetime.now(EST).strftime('%Y-%m-%d')
        qs=f"Update Tickers Set Trained_Date='{currentdate}',trained_filename='{self.model_name}', TrainingParams = '{trainingparams}'  Where Symbol = '{tkr}'"
        self.cursor.execute(qs)
        self.cnxn.commit()
        self.cnxn.close()


class predictmodel():
    def __init__(self,tkr=None,data_df=None):
        self.model_name = f"{tkr}-{LOSS}-{OPTIMIZER}-{CELL.__name__}-seq-{N_STEPS}-step-{LOOKUP_STEP}-layers-{N_LAYERS}-units-{UNITS}"
        self.data_df=data_df
        self.tkr = tkr

    def get_accuracy(self,model, data):
        y_test = data["y_test"]
        X_test = data["X_test"]
        y_pred = model.predict(X_test)
        y_test = np.squeeze(data["column_scaler"]["FutureValue"].inverse_transform(np.expand_dims(y_test, axis=0)))
        y_pred = np.squeeze(data["column_scaler"]["FutureValue"].inverse_transform(y_pred))
        y_pred = list(map(lambda current, future: int(float(future) > float(current)), y_test[:-LOOKUP_STEP], y_pred[LOOKUP_STEP:]))
        y_test = list(map(lambda current, future: int(float(future) > float(current)), y_test[:-LOOKUP_STEP], y_test[LOOKUP_STEP:]))
        return accuracy_score(y_test, y_pred)
        
    def predict(self,model, data):
        # retrieve the last sequence from data
        last_sequence = data["last_sequence"][-N_STEPS:]
        # retrieve the column scalers
        column_scaler = data["column_scaler"]
        # reshape the last sequence
        last_sequence = last_sequence.reshape((last_sequence.shape[1], last_sequence.shape[0]))
        # expand dimension
        last_sequence = np.expand_dims(last_sequence, axis=0)
        # get the prediction (scaled from 0 to 1)
        prediction = model.predict(last_sequence)
        # get the price (by inverting the scaling)
        predicted_price = column_scaler["FutureValue"].inverse_transform(prediction)[0][0]
        return predicted_price

    def testmodel(self):
        # load the data
        self.modelclass = buildmodel(self.tkr,self.data_df)

        data = self.modelclass.load_data(n_steps=N_STEPS, test_size=TEST_SIZE,
                        shuffle=False)
        # construct the model
        # model = self.modelclass.create_model(N_STEPS, loss=LOSS, units=UNITS, cell=CELL, n_layers=N_LAYERS,
        #                     dropout=DROPOUT, optimizer=OPTIMIZER, bidirectional=BIDIRECTIONAL)
        
        blob = BlobClient.from_connection_string(conn_str=os.environ.get('blob_conn_str'), container_name="tensorflow", blob_name="results"+"/"+self.model_name + ".h5")

        with tempfile.TemporaryDirectory() as tmpdirname:
            logging.info(tmpdirname)
            with open(tmpdirname + "/" + self.model_name + ".h5", "wb") as my_blob:
                blob_data = blob.download_blob()
                blob_data.readinto(my_blob)
            model = tf.keras.models.load_model(tmpdirname + "/" + self.model_name + ".h5")

        # evaluate the model
        mse, mae = model.evaluate(data["X_test"], data["y_test"], verbose=0)
        # calculate the mean absolute error (inverse scaling)
        mean_absolute_error = data["column_scaler"]["FutureValue"].inverse_transform([[mae]])[0][0]

        # predict the future price
        future_price = self.predict(model, data)
        accuracy_score=self.get_accuracy(model, data)
        # print(f"Future price after {LOOKUP_STEP} days is {future_price:.2f}$")
        conn_str='DRIVER={ODBC Driver 17 for SQL Server};SERVER='+os.environ.get('server')+ \
            ';DATABASE='+os.environ.get('database')+ \
                ';UID='+os.environ.get('dbusername')+ \
                    ';PWD='+ os.environ.get('dbpassword')
        self.cnxn = pyodbc.connect(conn_str)       
        self.cursor = self.cnxn.cursor()
        self.cursor.execute(f"Update Tickers Set Predicted_Inc={future_price},Model_Accuracy={accuracy_score}  Where Symbol = '{tkr}'")
        self.cnxn.commit()
        self.cnxn.close()

        return accuracy_score,future_price



def main(req: func.HttpRequest) -> None:
    logging.info('Python HTTP trigger function processed a request.')
    hasdata = req.params.get('name')
    if not hasdata:
        return 
    global tkr
    global data_df
    ticker = req.params.get('name')
    todo = req.params.get('Train')
    for tkr in ticker.split(','):
        datablob = BlobClient.from_connection_string(conn_str=os.environ.get('blob_conn_str'), container_name="tensorflow", blob_name=f"{tkr}.csv")
        with tempfile.TemporaryDirectory() as tmpdirname:
            with open(tmpdirname + "/" + tkr + ".csv", "wb") as mydata:
                blob_data = datablob.download_blob()
                blob_data.readinto(mydata)
                data_df = pd.read_csv(tmpdirname + "/" + tkr+'.csv')
                data_df=data_df.replace([np.inf, -np.inf], np.nan)
        datablob.delete_blob()
        cls=buildmodel(tkr,data_df)
        future = 0
        logging.info(f'Symbol:{tkr}, Train:{todo}')

        try:
            if todo=='True':
                cls.train_data()
                logging.info(f'Completed Training {tkr}')
            else:
                pred=predictmodel(tkr,data_df)
                accuracy,future = pred.testmodel()
                logging.info(str(accuracy) + ' ' + str(future))
        except:
            logging.info(f'{tkr} Failed to Train/Test')
            continue
    

global tkr
global data_df
np.random.seed(314)
tf.random.set_seed(314)
random.seed(314)



# Window size or the sequence length
N_STEPS = 100
# Lookup step, 1 is the next period of time
LOOKUP_STEP = 15

# test ratio size, 0.2 is 20%
TEST_SIZE = 0.4
# features to use
FEATURE_COLUMNS = ['High','Low','Open','Close','Volume']
# date now
date_now = time.strftime("%Y-%m-%d")

### model parameters

N_LAYERS = 3
# LSTM cell
CELL = LSTM
# 256 LSTM neurons
UNITS = 300
# 40% dropout
DROPOUT = 0.2
# whether to use bidirectional RNNs
BIDIRECTIONAL = False

### training parameters

# mean absolute error loss
# LOSS = "mae"
# huber loss
LOSS = "huber_loss"
OPTIMIZER = "adam"
BATCH_SIZE = 64
EPOCHS = 15
