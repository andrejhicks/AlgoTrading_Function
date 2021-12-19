import logging
from dotenv import load_dotenv
load_dotenv()
import azure.functions as func
import os
import json
import math
import time
import datetime as dt
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
from tensorflow import random as tf_random
from tensorflow.keras.models import load_model,Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from azure.storage.blob import BlobClient
from sklearn.metrics import accuracy_score
try:
    import tempfile
except:
    pass


"""
Modified source code from tutorial https://www.thepythoncode.com/article/stock-price-prediction-in-python-using-tensorflow-2-and-keras to create, train, test a model  using Tensorflow and Keras
"""
class predictmodel():
    def __init__(self,tkr=None,data_df=None):
        self.params = json.loads(ticker_detail.loc[tkr,'Params'])
        self.model_name = f"""{tkr}-{self.params["LOSS"]}-{self.params["OPTIMIZER"]}-{CELL.__name__}-seq-{self.params["N_STEPS"]}-step-{self.params["LOOKUP_STEP"]}-layers-{self.params["N_LAYERS"]}-units-{self.params["UNITS"]}"""
        self.data_df=data_df
        self.tkr = tkr
    def load_data(self, n_steps=50, scale=True, shuffle=True,test_size=0.2):
        """
        Imports data from CSV file created in the intradayanalysis.py file and stored in blob storage. 
        """
        lookup_step=self.params["LOOKUP_STEP"]
               
        futureval=(self.data_df['Close']-self.data_df['Close'].shift(lookup_step))/self.data_df['Close'].shift(lookup_step)
        self.data_df.loc[lookup_step:,'FutureValue']=futureval
        
        #Add columns to the feature columns. Default is a minimal set of OHLC only. All columns in the dataframe will be added to this list
        global FEATURE_COLUMNS
        FEATURE_COLUMNS=[]
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

    def get_accuracy(self,model, data):
        y_test = data["y_test"]
        X_test = data["X_test"]
        y_pred = model.predict(X_test)
        y_test = np.squeeze(data["column_scaler"]["FutureValue"].inverse_transform(np.expand_dims(y_test, axis=0)))
        y_pred = np.squeeze(data["column_scaler"]["FutureValue"].inverse_transform(y_pred))
        y_pred = list(map(lambda current, future: int(float(future) > float(current)), y_test[:-self.params["LOOKUP_STEP"]], y_pred[self.params["LOOKUP_STEP"]:]))
        y_test = list(map(lambda current, future: int(float(future) > float(current)), y_test[:-self.params["LOOKUP_STEP"]], y_test[self.params["LOOKUP_STEP"]:]))
        return accuracy_score(y_test, y_pred)
        
    def predict(self,model, data):
        # retrieve the last sequence from data
        last_sequence = data["last_sequence"][-self.params["N_STEPS"]:]
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

        data = self.load_data(n_steps=self.params["N_STEPS"], test_size=self.params["TEST_SIZE"],
                        shuffle=False)

        blob = BlobClient.from_connection_string(conn_str=os.environ.get('blob_conn_str'), container_name="tensorflow", blob_name="results"+"/"+self.model_name + ".h5")
        t1 = datetime.now()
        with tempfile.TemporaryDirectory() as tmpdirname:
            logging.info(tmpdirname)
            with open(tmpdirname + "/" + self.model_name + ".h5", "wb") as my_blob:
                blob_data = blob.download_blob()
                blob_data.readinto(my_blob)
            model = load_model(tmpdirname + "/" + self.model_name + ".h5")
        logging.info(datetime.now()-t1)
        model.save_weights('testckpt{}.cpt'.format(tkr))
        del model
        model = self.create_model(self.params["N_STEPS"], loss=self.params["LOSS"], units=self.params["UNITS"], cell=CELL, n_layers=self.params["N_LAYERS"],
                                    dropout=self.params["DROPOUT"], optimizer=self.params["OPTIMIZER"], bidirectional=self.params["BIDIRECTIONAL"])
        model.load_weights('testckpt{}.cpt'.format(tkr))
        
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
        self.cursor.execute(f"Update Tickers Set Predicted_Inc={future_price},Model_Accuracy={accuracy_score},Modified_Date=GETDATE() Where Symbol = '{tkr}'")
        self.cnxn.commit()
        self.cnxn.close()

        return accuracy_score,future_price

class npanalysis():
    def __init__(self):
        ##Set parameters for the ranking session based on backtesting results
        self.conn_str='DRIVER={ODBC Driver 17 for SQL Server};SERVER='+os.environ.get('server')+ \
            ';DATABASE='+os.environ.get('database')+ \
                ';UID='+os.environ.get('dbusername')+ \
                    ';PWD='+ os.environ.get('dbpassword')
        

    def importdata(self):
        self.cnxn = pyodbc.connect(self.conn_str)       
        self.cursor = self.cnxn.cursor()
        query = "Select Distinct Ticker, DateIndex, High, Low, Open_, Close_, Volume From HourData Where DateIndex>DateAdd(Month,-6,GetDate()) and Ticker in ('{}')".format("','".join(tickers))
        self.cursor.execute(query)
        dataquery1 = [list(ele) for ele in self.cursor]
        self.cnxn.close()
        daily_df = pd.DataFrame(dataquery1,columns=['Ticker','Date','High','Low','Open','Close','Volume'])
        daily_df['Date']=pd.to_datetime(daily_df['Date'])
        lastestdate = daily_df['Date'].max()
        recenttickers=daily_df.loc[daily_df['Date']==lastestdate,'Ticker'].tolist()
        daily_df=daily_df[daily_df['Ticker'].isin(recenttickers)]
        daily_df['DateInt']=pd.to_numeric(daily_df['Date'].dt.strftime('%Y%m%d%H%M'))
        daily_df=daily_df.pivot(index='Date',columns='Ticker',values=['High','Low','Open','Close','Volume','DateInt'])
        daily_df.sort_index(inplace=True)
        # Get all tickers in the daily_df dataframe in the correct order
        self.tickers=np.unique(daily_df['High'].columns.to_numpy(copy=True))
        rows=len(daily_df.index)
        #Create a 3D array for calculation of indicators
        self.daily_np=np.empty(shape=(rows,6,self.tickers.shape[0]))
        self.daily_df=daily_df.copy()
        idx=pd.IndexSlice
        #Build the numpy array by looping through the array and placing the array with the z-axis being ticker
        for tn,t in enumerate(self.tickers):
            self.daily_np[:,:,tn]=self.daily_df.loc[:,idx[:,t]].to_numpy(copy=True)
        np.nan_to_num(self.daily_np,copy=True)
            
        self.y=self.daily_np.shape[0]
        self.x=self.daily_np.shape[1]
        self.z=self.daily_np.shape[2]
        print('The size of the data analysis X:{} Rows, Y:{} Columns, Z:{} Indicators'.format(str(self.y),str(self.x),str(self.z)))

    def bollinger(self):
        bollinger=np.empty(shape=(self.y,2,self.z))
        window=70
        st=dt.datetime.now()
        #loop through each ticker and calculate the SMA and 1x Standard deviation. This can be multiplied as needed for bollinger limits
        for t in range(self.z):
            stddev=np.empty(0)
            bollinger[int(window-1):,0,t]=np.convolve(self.daily_np[:,3,t],np.ones(window),'valid')/window
            for n in range(window-1,self.y):
                bollinger[n,1,t]=np.std(self.daily_np[n-window:n,3,t])
        print(dt.datetime.now()-st)
        return bollinger

    def fisher_transform(self,period=50,days=800):
        fisher=np.zeros(shape=(self.y,1,self.z))
        r=self.y
        Xax=np.arange(0,days,1)
        for d in range(days,-1,-1):
            f=self.daily_np[r-period-d:r-d,3,:]
            # print(f.shape)
            minf=np.amin(f,0).flatten()
            maxf=np.amax(f,0).flatten()
            diff_f=maxf-minf
            scalarf=(f-minf)/diff_f*2-1
            scalarf=np.where(scalarf==-1,-.999,scalarf)
            scalarf=np.where(scalarf==1,.999,scalarf)
            for i in range(self.z):
                # print(i)
                fisher[-d,0,i]=.5*math.log(np.divide((1+scalarf[-1:,i]),(1-scalarf[-1:,i])))+.5*fisher[-d-1,0,i]
        return fisher
    
    def CalcRSI(self,rsiperiod):
        # Calculate RSI
        # rsiperiod=14
        # prev_close=np.vstack((np.zeros((1,self.close_np.shape[1])),self.close_np[:-1,:]))
        diffclose=np.zeros(shape=(self.y-1,1,self.z))
        for t in range(self.z):
            col = self.daily_np[:,3,t]
            diffclose[:,:,t]=np.vstack(np.diff(col,axis=0))
        gain=np.where(diffclose>=0,diffclose,0)
        loss=np.abs(np.where(diffclose<0,diffclose,0))
        gainavg=np.zeros(gain.shape)
        lossavg=np.zeros(loss.shape)
        gainavg[rsiperiod-1,:]=np.mean(gain[:rsiperiod-1,:],axis=0)
        lossavg[rsiperiod-1,:]=np.mean(loss[:rsiperiod-1,:],axis=0)
        for r in range(rsiperiod,gainavg.shape[0]):
            gainavg[r,:]=(gain[r,:]+gainavg[r-1,:]*(rsiperiod-1))/rsiperiod
            lossavg[r,:]=(loss[r,:]+lossavg[r-1,:]*(rsiperiod-1))/rsiperiod
        # relstrength=np.divide(gainavg,lossavg,where=(lossavg>0))

        RSI=np.append(np.zeros(shape=(1,1,self.z)),-100/(np.divide(gainavg,lossavg,where=(lossavg>0))+1)+100,axis=0)
        return RSI

    def linear_regression(self,scaled_data,period,starting = 0):
        #by default only perform linear regression on the last data point
        return np.polyfit(range(period),scaled_data,deg=1)

    def createindicators(self):
        
        self.importdata()
        print('bollinger')
        Bollinger =self.bollinger()
        print('Fisher')
        Fisher_Transform = self.fisher_transform()
        print('RSI')
        RSI = self.CalcRSI(10)
        print('linreg')
        # linreg=np.empty()
        blob = [None]*self.tickers.shape[0]
        trends = [13,10*13,20*13]
        numcols = int(len(trends)*2)
        print(self.daily_np.shape)
        LinReg=np.zeros(shape=(self.y,numcols,self.z),dtype=float)
        min_max_scaler = preprocessing.MinMaxScaler()
        for c,p in enumerate(trends):
            print(p)
            for s in range(self.y-p):
                r=self.y-s
                sclar_data = min_max_scaler.fit_transform(self.daily_np[r-p:r,3,:])
                a=self.linear_regression(sclar_data,period=p,starting=s)
                LinReg[self.y-s-1,c,:]=a[0]
            LinReg[self.y-s-1,c+2,:]=a[1]
        
        idx=pd.IndexSlice
        self.df = pd.DataFrame()
        for t,ticker in enumerate(tickers):
            try:
                df = self.daily_df.loc[:,idx[:,ticker]]
            except:
                continue
            df.columns=df.columns.droplevel(1)
            df.loc[df.index <= max(df.index)-dt.timedelta(days=10)]
            df['date'] = df.index
            df['hour'] = df['date'].dt.hour
            df['dayofweek'] = df['date'].dt.dayofweek
            df['quarter'] = df['date'].dt.quarter
            df['month'] = df['date'].dt.month
            df['year'] = df['date'].dt.year
            df['dayofyear'] = df['date'].dt.dayofyear
            df['dayofmonth'] = df['date'].dt.day
            # df.loc[True,'weekofyear'] = df['date'].dt.weekofyear
            df.drop('date',axis=1,inplace=True)
            df.loc[:,'SMA']=Bollinger[:,0,t]
            # df['PosStddev']=Bollinger[:,0,t]+Bollinger[:,1,t]*2
            # df['NegStddev']=Bollinger[:,0,t]-Bollinger[:,1,t]*2
            df.loc[:,'FisherTransform']=Fisher_Transform[:,0,t]
            df.loc[:,'RSI']=RSI[:,0,t]
            # df['FutureClose']=df['Close'].shift(-10)
            for col,interval in enumerate(trends):
                df.loc[:,'LinRegSlope'+str(interval)]=LinReg[:,col,t]
                # df['LinRegInt'+str(interval)]=LinReg[:,col+2,t]
            df=df[df['SMA']!=0]
            df['Symbol']=ticker
            self.df=pd.concat([self.df,df])
            logging.info(self.df.head())
        return self.df

def test():# main(req: func.HttpRequest) -> None:
    global tickers
    global ticker_detail
    logging.info('Python HTTP trigger function processed a request.')
    # hasdata = req.params.get('name')
    # if not hasdata:
    #     return 
    global tkr
    global data_df
    # ticker = req.params.get('name')
    # tickers = ticker.split(',')
    tickers=['A','AAPL']
    alldf = npanalysis().createindicators()
    cnxn=pyodbc.connect(npanalysis().conn_str)
    cursor=cnxn.cursor()
    cursor.execute("Select Symbol, trained_filename, TrainingParams From Tickers Where Symbol in ('{}')".format("','".join(tickers)))
    ticker_detail=pd.DataFrame([list(ele) for ele in cursor], columns=["Symbol","Filename","Params"])
    ticker_detail.set_index('Symbol',inplace=True)
    cnxn.close()
    for tkr in tickers:
        data_df=alldf[alldf['Symbol']==tkr].copy()
        data_df.drop(['Symbol'],inplace=True,axis=1)
        future = 0
        logging.info(f'Symbol:{tkr}')

        # try:
        pred=predictmodel(tkr,data_df)
        accuracy,future = pred.testmodel()
        logging.info(str(accuracy) + ' ' + str(future))
        print(str(accuracy) + ' ' + str(future))
        # except:
        #     logging.info(f'{tkr} Failed to Predict')
        #     continue
    return

global tkr
global data_df
global ticker_detail
np.random.seed(314)
tf_random.set_seed(314)
random.seed(314)
CELL = LSTM

tickers=[]
test()