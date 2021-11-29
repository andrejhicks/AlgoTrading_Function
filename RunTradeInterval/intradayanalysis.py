import time
from datetime import datetime
import datetime as dt
import pytz
EST=pytz.timezone('US/Eastern')
from dotenv import load_dotenv
import logging
import concurrent.futures
import math
import os
import pandas as pd
import numpy as np
from sklearn import preprocessing
import pyodbc
import requests
from azure.storage.blob import BlobClient
load_dotenv()
import tempfile
import azure.functions as func

#non-production libraries
# from xgboost import plot_importance, plot_tree
# import plotly.graph_objs as go
# from plotly.subplots import make_subplots
# import matplotlib.pyplot as plt
# import testtensorflow as tsttf

passkey='prod'                   
idx=pd.IndexSlice
global dd

class importmarketdata():
    def __init__(self):
        conn_str='DRIVER={ODBC Driver 17 for SQL Server};SERVER='+os.environ.get('server')+ \
            ';DATABASE='+os.environ.get('database')+ \
                ';UID='+os.environ.get('dbusername')+ \
                    ';PWD='+ os.environ.get('dbpassword')
        # self.uri=self.credintials.logicappuri
        starttime=datetime.strptime('09:00','%H:%M')
        tdelta=dt.timedelta(minutes=30)
        self.time_array=[]
        for i in range(13):
            self.time_array.append(str(starttime+tdelta*i))
            ##Set parameters for the ranking session based on backtesting results
        self.cnxn = pyodbc.connect(conn_str)       
        self.cursor = self.cnxn.cursor()
        self.cursor.execute("SELECT Symbol FROM Tickers")
        tickerssql = self.cursor.fetchall()
        self.tickers=[]
        for ticker in tickerssql:
            ticker = ticker[0]
            self.tickers.append(ticker)
        self.tickers.sort()
        self.tickers=self.tickers
        if passkey=='prod':
            self.hourtbl = 'HourData'
        else:
            self.hourtbl = 'HourData'
    def import_iex_data(self,requesturl,requestparams):       
        retry=0
        while True and retry<3:
            try:
                resp=requests.get(requesturl,params=requestparams)
                d=resp.json()
                df = pd.concat([pd.DataFrame(v) for k,v in d.items()], keys=d)
            except:
                retry+=1
                continue
            break
        try:
            df=df['chart'].apply(pd.Series)
        except:
            pass
        self.load_iex=pd.concat([self.load_iex,df])

    def update_intraday_iex(self):
        print('Running Data Retrieval in ' + passkey)
        run=0
        df = pd.DataFrame()
        day_delta = dt.timedelta(days=1)
        self.cursor.execute(f"SELECT Max(DateIndex) FROM {self.hourtbl}")
        self.startdate=self.cursor.fetchone()[0]
        day_delta = dt.timedelta(days=1)
        self.startdate=self.startdate+day_delta
        print(self.startdate)
        if passkey=='test':
            self.startdate=self.startdate#datetime.strptime(self.startdate,'%Y-%m-%d').date()
            token=os.environ.get('IEXTestKey')
            base_url = 'https://sandbox.iexapis.com/stable'
        else:
            self.startdate=self.startdate+day_delta#datetime.strptime(self.startdate,'%Y-%m-%d').date()+day_delta
            token=os.environ.get('IEXProdKey')
            base_url = 'https://cloud.iexapis.com/v1'
        
        tickerdf=pd.DataFrame()        
        interval=390
        tickerdata=pd.DataFrame()
        end_date=dt.date.today()+day_delta
        s=self.startdate.strftime('%Y-%m-%d 00:00:00')
        f=self.startdate.strftime('%Y-%m-%d 16:00:00')
        # self.tickers=self.tickers[:120]
        if self.startdate.date==datetime.now(EST).date:
            qs=f"Select Ticker, Count(Ticker) From HourData Where DateIndex > '{s}' and DateIndex <= '{f}' Group By Ticker"
            self.cursor.execute(qs)
            latestdate = self.cursor.fetchall()
            chartrecords = pd.DataFrame.from_records(latestdate,columns=['Ticker','NumRecords'])
            chartrecords.set_index('Ticker',inplace=True)

        for run,i in enumerate(range((end_date - self.startdate.date()).days)):
            self.starttime=datetime.strptime((self.startdate+i*day_delta).strftime('%Y-%m-%d 08:30:00'),'%Y-%m-%d %H:%M:%S')
            timefromopen = datetime.strptime(datetime.now(EST).strftime('%Y-%m-%d %H:%M:%S'),'%Y-%m-%d %H:%M:%S')-datetime.strptime(datetime.now(EST).strftime('%Y-%m-%d 09:30:00'),'%Y-%m-%d %H:%M:%S')
            timefromopen = round(timefromopen.total_seconds()/60,0)

            self.load_iex=pd.DataFrame()
            tradedate=self.startdate + i*day_delta
            # if not tc.get_calendar("XNYS").is_session(pd.Timestamp(tradedate.strftime('%Y-%m-%d'))):
            #     continue
            print('Stock Data Date: ' + str(tradedate.strftime('%Y-%m-%d')))
            logging.info('Stock Data Date: ' + str(tradedate.strftime('%Y-%m-%d')))
            logging.info('Chart Last: {}'.format(str(timefromopen)))
            exdate=tradedate.strftime('%Y%m%d')
            evaltickers=""
            df = pd.DataFrame()
            tickerdf=pd.DataFrame() 
            request_array = []
            urllist = []
            for t,ticker in enumerate(self.tickers,1):
                evaltickers=evaltickers+","+ticker
                if self.startdate.date==datetime.now(EST).date:
                    try:
                        charted=chartrecords.at[ticker,'NumRecords']
                    except:
                        charted=400
                    if charted == 0 or run!=0:
                        cl = 400
                    else:
                        cl = timefromopen-charted*30
                else:
                    cl=400
                if t%90==0:
                    params={'token': token,\
                            'symbols':evaltickers[1:], \
                            'types':'chart',\
                            'exactDate':str(exdate), \
                            'chartIEXOnly':'false', \
                            'chartLast': cl
                            }
                    urllist.append(base_url+'/stock/market/batch')
                    request_array.append(params)
                    evaltickers=""
            retry=0
            if len(evaltickers)>2:
                params={'token': token,\
                        'symbols':evaltickers[1:], \
                        'types':'chart',\
                        'exactDate':str(exdate),
                        'chartIEXOnly':'false', \
                        'chartLast': cl
                        }
                urllist.append(base_url+'/stock/market/batch')
                request_array.append(params)

            with concurrent.futures.ThreadPoolExecutor(6) as executor:
                executor.map(self.import_iex_data,urllist,request_array)
            load_iex=self.load_iex
            print(load_iex.shape)
            del self.load_iex
            load_iex.reset_index(inplace=True)
            if load_iex.empty:
                continue
            load_iex['DateTime']=load_iex['date']+' '+load_iex['minute']

            load_iex['DateTime']=pd.to_datetime(load_iex['DateTime'])-dt.timedelta(hours=1)
            load_iex.rename(columns={'level_0':'Ticker'},inplace=True)
            load_iex.set_index(['Ticker','DateTime'],inplace=True)
            time_step=dt.timedelta(minutes=30)

            for i in self.tickers:
                try:
                    df=load_iex.loc[idx[i,:],:].copy()
                except:
                    continue
          
                df.sort_values(by='minute',ascending=True,inplace=True)
                for t in range(math.floor(interval/30)):
                    a=self.starttime+t*time_step
                    b=self.starttime+(t+1)*time_step#+dt.timedelta(minutes=1)
                    avg_vals=df.loc[idx[:,a:b],:]

                    openprice=avg_vals.loc[idx[:,:],'marketOpen'].to_numpy(copy=True)
                    closeprice=avg_vals.loc[idx[:,:],'marketClose'].to_numpy(copy=True)
                    if openprice.shape[0]<20:
                        continue
                    o=0
                    c=0
                    for chk in range(20):
                        if o==0:
                            if openprice[chk]>0:
                                o=openprice[chk]
                        if c==0:
                            if closeprice[-chk]>0:
                                c=closeprice[-chk]                           
                    timerec=b

                    new_data=pd.DataFrame([[i,timerec,avg_vals['marketHigh'].max(), \
                        avg_vals['marketLow'].min(),o,c,avg_vals['marketVolume'].sum()]], \
                        columns=['Ticker','DateIndex','high','low','open','close','volume'])
                    tickerdf=pd.concat([tickerdf,new_data],ignore_index=True)

            tickerdf.fillna(0,inplace=True)
            
            query=f"INSERT INTO {self.hourtbl} (Ticker,DateIndex,High,Low,Close_,Open_,Volume) VALUES "
            queryarray=[]
            if tickerdf.empty:
                print('No Data for ' + str(tradedate.strftime('%Y-%m-%d')))
                continue
            for count,t in enumerate(tickerdf.itertuples(),1):
                # print(t)
                query=query + """('{}','{}',{},{},{},{},{}),""".format(t.Ticker,t.DateIndex,t.high,t.low,t.open,t.close,t.volume)

                if count%1000==0 and count>0:
                    queryarray.append(query[:-1])
                    query=f"INSERT INTO {self.hourtbl} (Ticker,DateIndex,High,Low,Close_,Open_,Volume) VALUES "
            queryarray.append(query[:-1])
            if len(queryarray)>0:
                for count,q in enumerate(queryarray):
                    self.cursor.execute(q.replace('nan','0'))
                    if passkey=='prod':
                        self.cnxn.commit()

        return #tickerdata

class npanalysis():
    def __init__(self):
        ##Set parameters for the ranking session based on backtesting results
        self.cnxn = importmarketdata().cnxn   
        self.cursor = importmarketdata().cursor

    def importdata(self):

        query = "Select Distinct Ticker, DateIndex, High, Low, Open_, Close_, Volume From HourData Where DateIndex>DateAdd(Month,-6,GetDate())"# and Ticker in ('NKE','CVX','TSLA', 'FB')"
        self.cursor.execute(query)
        dataquery1 = [list(ele) for ele in self.cursor]
        daily_df = pd.DataFrame(dataquery1,columns=['Ticker','Date','High','Low','Open','Close','Volume'])
        dataquery1=[]
        daily_df['Date']=pd.to_datetime(daily_df['Date'])
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

def create_model():       
    mkt=npanalysis()
    mkt.importdata()
    marketdata = mkt.daily_df
    print('bollinger')
    Bollinger =mkt.bollinger()
    print('Fisher')
    Fisher_Transform = mkt.fisher_transform()
    print('RSI')
    RSI = mkt.CalcRSI(10)
    print('linreg')
    # linreg=np.empty()
    
    trends = [13,10*13,20*13]
    numcols = int(len(trends)*2)
    print(mkt.daily_np.shape)
    LinReg=np.zeros(shape=(mkt.y,numcols,mkt.z),dtype=float)
    min_max_scaler = preprocessing.MinMaxScaler()
    for c,p in enumerate(trends):
        print(p)
        for s in range(mkt.y-p):
            r=mkt.y-s
            sclar_data = min_max_scaler.fit_transform(mkt.daily_np[r-p:r,3,:])
            a=mkt.linear_regression(sclar_data,period=p,starting=s)
            LinReg[mkt.y-s-1,c,:]=a[0]
            LinReg[mkt.y-s-1,c+2,:]=a[1]
    traintickers = ''
    testtickers = ''
    funcurl = os.environ.get('FunctionURL')
    funckey = os.environ.get('FunctionKey')
    for t,ticker in enumerate(mkt.tickers[25:]):
        df=createtable(mkt.daily_df,ticker)
        
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
        # df.to_csv(f'DataFiles/{ticker}.csv',index=False)
        blob = BlobClient.from_connection_string(conn_str=os.environ.get('blob_conn_str'), container_name="tensorflow", blob_name=ticker + ".csv")
        if blob.exists():
            blob.delete_blob()            
        with tempfile.TemporaryDirectory() as tmpdirname:
            df.to_csv(f'{tmpdirname}/{ticker}.csv',index=False)
            with open(tmpdirname + "/" + ticker + ".csv", "rb") as my_blob:
                blob.upload_blob(my_blob)
        
        mkt.cursor.execute(f"Select Trained_Date,trained_filename From Tickers Where Symbol='{ticker}'")
        trained_model = mkt.cursor.fetchall()

        if trained_model[0][0]==None:
            sincetraining=100
        else:
            sincetraining = (trained_model[0][0]-datetime.now().date()).days
        if sincetraining>45 or trained_model[0][1]==None:
            if traintickers!='': 
                traintickers = str(traintickers+ ','+ticker) 
            else:
               traintickers = ticker
        else:
            if testtickers!='':
                testtickers = str(testtickers+ ',' +ticker)  
            else: 
                testtickers = ticker

        if (t%10==0 and t!=0) or t==mkt.tickers.shape[0]-1:
            # if sincetraining>45 or trained_model[0][1]==None:
            if passkey == 'prod' and traintickers!='':
                print(f'{funcurl}?name={traintickers}&Train=True&code={funckey}==')
                try:
                    requests.post(f'{funcurl}?name={traintickers}&Train=True&code={funckey}==',timeout=1)
                except:
                    logging.info(f'Failed to Process Training {traintickers}')
                    print(f'Failed to Process Training {traintickers}')            
            else:
                print(f'Info Only == {funcurl}?name={traintickers}&Train=True&code={funckey}==')

            if passkey == 'prod' and testtickers!='':
                print(f'{funcurl}?name={testtickers}&Train=False&code={funckey}==')
                try:
                    requests.post(f'{funcurl}?name={testtickers}&Train=False&code={funckey}==',timeout=1)
                except:
                    logging.info(f'Failed to Process Testing {testtickers}')
                    print(f'Failed to Process Testing {testtickers}')
            else:
                print(f'Info Only == {funcurl}?name={testtickers}&Train=False&code={funckey}==')
            traintickers = ''
            testtickers = ''
                

def createtable(df,ticker):
    df = df.loc[:,idx[:,ticker]]
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
    
    return df

def main(mytimer: func.TimerRequest) -> None:
    funcurl = os.environ.get('FunctionURL')
    funckey = os.environ.get('FunctionKey')
    requests.post(f'{funcurl}?&code={funckey}==',timeout=1)
    currtime=datetime.now(EST)
    logging.info(currtime)
    hour = currtime.hour
    min = currtime.min
    if (hour>=15 and min > 25):
        return
        #Initialize Brokerage API conncection
    elif requests.get('https://cloud.iexapis.com/stable/stock/twtr/quote?token={}'.format(os.environ.get("IEXProdKey"))).json()['isUSMarketOpen'] or (hour==15 and min < 25):
        dd=importmarketdata()
        dd.update_intraday_iex()
        create_model()
    else:
        print('Market Closed')
    #Wait before executing trades
    time.sleep(60000)
    dd.cursor.execute("Select UserKey From TradeServiceUsers Where Active = 1")
    users=[list(ele)[0] for ele in dd.cursor]
    funcurl = os.environ.get('TradeFunctionUrl') 
    funckey = os.environ.get('TradeFunctionKey')
    for user in users:
        requests.post(f'{funcurl}?name={user}&code={funckey}==',timeout=.01)
    dd.cnxn.close()

# passkey='test'
dd=importmarketdata()
# dd.update_intraday_iex()
create_model()
