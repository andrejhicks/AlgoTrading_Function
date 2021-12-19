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
import pyodbc
import requests
load_dotenv()
import azure.functions as func

passkey='prod'                   
idx=pd.IndexSlice
global dd

class importmarketdata():
    def __init__(self):
        conn_str='DRIVER={ODBC Driver 17 for SQL Server};SERVER='+os.environ.get('server')+ \
            ';DATABASE='+os.environ.get('database')+ \
                ';UID='+os.environ.get('dbusername')+ \
                    ';PWD='+ os.environ.get('dbpassword')
        
        starttime=datetime.strptime('09:00','%H:%M')
        tdelta=dt.timedelta(minutes=30)
        self.time_array=[]
        for i in range(13):
            self.time_array.append(str(starttime+tdelta*i))
            ##Set parameters for the ranking session based on backtesting results
        tries=0
        self.tickers=[]
        while tries<5 and self.tickers==[]:
            try:
                self.cnxn = pyodbc.connect(conn_str)       
                self.cursor = self.cnxn.cursor()
                self.cursor.execute("SELECT Symbol FROM Tickers")
                tickerssql = self.cursor.fetchall()
                for ticker in tickerssql:
                    ticker = ticker[0]
                    self.tickers.append(ticker)
                self.tickers.sort()
                self.tickers=self.tickers
            except:
                tries+=1
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
            self.startdate=self.startdate#datetime.strptime(self.startdate,'%Y-%m-%d').date()+day_delta
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
                if t%50==0:
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
            load_iex=self.load_iex.copy()
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
                if df['notional'].sum(skipna=True)==0:
                    continue
                df.sort_values(by='minute',ascending=True,inplace=True)
                for t in range(math.floor(interval/30)):
                    a=self.starttime+t*time_step
                    b=self.starttime+(t+1)*time_step#+dt.timedelta(minutes=1)
                    avg_vals=df.loc[idx[:,a:b],:]

                    openprice=avg_vals.loc[idx[:,:],'average'].to_numpy(copy=True)
                    closeprice=avg_vals.loc[idx[:,:],'average'].to_numpy(copy=True)
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

                    new_data=pd.DataFrame([[i,timerec,avg_vals['high'].max(), \
                        avg_vals['low'].min(),o,c,avg_vals['notional'].sum()]], \
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

                if count%500==0 and count>0:
                    queryarray.append(query[:-1])
                    query=f"INSERT INTO {self.hourtbl} (Ticker,DateIndex,High,Low,Close_,Open_,Volume) VALUES "
            queryarray.append(query[:-1])
            if len(queryarray)>0:
                for count,q in enumerate(queryarray):
                    self.cursor.execute(q.replace('nan','0'))
            if passkey=='prod':
                self.cnxn.commit()
            
        return #tickerdata

#Function to handle concurrent calls to training function
def call_train_test(Uri_request):
    try:
        print(Uri_request)
        logging.info(Uri_request)
        requests.post(Uri_request,timeout=0.01)
    except:
        logging.info(f'Failed to Process Training {Uri_request[73:100]}')

def create_model():
    logging.info("Generating Models")
    conn_str='DRIVER={ODBC Driver 17 for SQL Server};SERVER='+os.environ.get('server')+ \
            ';DATABASE='+os.environ.get('database')+ \
                ';UID='+os.environ.get('dbusername')+ \
                    ';PWD='+ os.environ.get('dbpassword') 
    retry=0
    while retry<=3:
        try:
            cnxn = pyodbc.connect(conn_str)
            break
        except:
            retry+=1
            continue  
    cursor = cnxn.cursor()

    cursor.execute("""Select Distinct Ticker From HourData Where DateIndex>DateAdd(Day,-20,GetDate())""")
    tickers = [t[0] for t in cursor]
    traintickers = ''
    testtickers = ''

    global funcurl
    global funckey
    funcurl = os.environ.get('FunctionURL')
    funckey = os.environ.get('FunctionKey')
    functioncalls=[]
    cursor.execute(f"Select Symbol,Trained_Date,trained_filename From Tickers Where Model_Accuracy>0.6")
    trainedmodels = [list(ele) for ele in cursor]
    trainedmodels = pd.DataFrame(trainedmodels,columns=['Ticker','Trained_Date','Filename'])
    trainedmodels.set_index('Ticker',inplace=True)
    logging.info("Running for {} Models".format(str(trainedmodels.shape[0])))

    for t,ticker in enumerate(trainedmodels.itertuples()):
        if ticker.Trained_Date==None:
            sincetraining=100
        else:
            sincetraining = (ticker.Trained_Date-datetime.now().date()).days
        if sincetraining>45 or ticker.Filename==None:
            if traintickers!='': 
                traintickers = str(traintickers+ ','+ticker.Index) 
            else:
               traintickers = ticker.Index
        else:
            if testtickers!='':
                testtickers = str(testtickers+ ',' +ticker.Index)  
            else: 
                testtickers = ticker.Index    
        lenfunc = len(testtickers.split(','))
        if (lenfunc%10==0 and lenfunc!=0) or t==trainedmodels.shape[0]-1:

            if passkey == 'prod' and len(traintickers)>0:
                pass
                # functioncalls.append(f'{funcurl}?name={traintickers}&Train=True&code={funckey}==')      
            else:
                print(f'Info Only == {funcurl}?name={traintickers}&Train=True&code={funckey}==')

            if passkey == 'prod' and len(testtickers)>0:
                functioncalls.append(f'{funcurl}?name={testtickers}&code={funckey}==')
            else:
                print(f'Info Only == {funcurl}?name={testtickers}&Train=False&code={funckey}==')
            traintickers = ''
            testtickers = ''

    with concurrent.futures.ThreadPoolExecutor(2) as executor:
        executor.map(call_train_test,functioncalls)
    cnxn.close()
    
def main(mytimer: func.TimerRequest) -> None:
    funcurl = os.environ.get('FunctionURL')
    funckey = os.environ.get('FunctionKey')

    currtime=datetime.now(EST)
    logging.info(currtime)
    hour = currtime.hour
    min = currtime.min
    if requests.get('https://cloud.iexapis.com/stable/stock/twtr/quote?token={}'.format(os.environ.get("IEXProdKey"))).json()['isUSMarketOpen'] or (hour==15 and min < 25):
        dd=importmarketdata()
        logging.info("Updating Data")
        dd.update_intraday_iex()
        logging.info("Finished Update, Moving to Models")
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
    return
