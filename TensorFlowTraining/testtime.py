import time
import datetime as dt
from datetime import datetime
import pytz
import pandas as pd
import numpy as np
import pyodbc
import os
from dotenv import load_dotenv
import concurrent.futures
import json
import requests
import requests
import pyodbc
import requests
from azure.storage.blob import BlobClient
import concurrent.futures
load_dotenv()
# Window size or the sequence length
EST=pytz.timezone('US/Eastern')
print(datetime.now(EST).strftime('%Y-%m-%d %H:%M:%S'))

exit()
def deleteblobs(ticker):
    blob = BlobClient.from_connection_string(conn_str=os.environ.get('blob_conn_str'), container_name="tensorflow", blob_name=ticker + ".csv")
    try:
        blob.delete_blob() 
        print('Delete ' +ticker)
    except:
        print('No ' +ticker)

conn_str='DRIVER={ODBC Driver 17 for SQL Server};SERVER='+os.environ.get('server')+ \
    ';DATABASE='+os.environ.get('database')+ \
        ';UID='+os.environ.get('dbusername')+ \
            ';PWD='+ os.environ.get('dbpassword')
# uri=credintials.logicappuri
starttime=datetime.strptime('09:00','%H:%M')
tdelta=dt.timedelta(minutes=30)
time_array=[]
for i in range(13):
    time_array.append(str(starttime+tdelta*i))
    ##Set parameters for the ranking session based on backtesting results
cnxn = pyodbc.connect(conn_str)       
cursor = cnxn.cursor()
cursor.execute("SELECT Symbol FROM Tickers")
tickerssql = cursor.fetchall()
tickers=[]
for ticker in tickerssql:
    ticker = ticker[0]
    tickers.append(ticker)
print('Starting Delete')
with concurrent.futures.ThreadPoolExecutor(6) as executor:
    executor.map(deleteblobs,tickers)
# for ticker in tickers:
#     blob = BlobClient.from_connection_string(conn_str=os.environ.get('blob_conn_str'), container_name="tensorflow", blob_name=ticker + ".csv")
#     try:
#         blob.delete_blob() 
#         print('Delete ' +ticker)
#     except:
#         print('No ' +ticker)
# exit()
funcurl = os.environ.get('FunctionURL')
funckey = os.environ.get('FunctionKey')
print(f'{funcurl}?code={funckey}==')
requests.post(f'{funcurl}?name=TSLA,FB&Train=True&code={funckey}==',timeout=50)

token=os.environ.get('IEXTestKey')
sybl='TWTR'
base_url = f'https://sandbox.iexapis.com/stable/stock/{sybl}/quote'
params={'token': token}
resp=requests.get(base_url,params=params)
d=resp.json()

print(d)
exit()

EST=pytz.timezone('US/Eastern')
print(datetime.now(EST).strftime('%Y-%m-%d %H:%M:%S'))

ticker='CVX'
conn_str='DRIVER={ODBC Driver 17 for SQL Server};SERVER='+os.environ.get('server')+ \
            ';DATABASE='+os.environ.get('database')+ \
                ';UID='+os.environ.get('dbusername')+ \
                    ';PWD='+ os.environ.get('dbpassword')
cnxn = pyodbc.connect(conn_str)       
cursor = cnxn.cursor()
cursor.execute(f"Select Trained_Date,trained_filename From Tickers Where Symbol='{ticker}'")
trained_model = cursor.fetchall()
# print((trained_model[0][0]-(datetime.now().date()-dt.timedelta(days=40))).days)#.strftime('%Y-%m-%d'))
print(trained_model[0][1]==None)