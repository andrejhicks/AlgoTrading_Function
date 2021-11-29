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

load_dotenv()
# Window size or the sequence length

funcurl = os.environ.get('FunctionURL')
funckey = os.environ.get('FunctionKey')
print(f'{funcurl}?code={funckey}==')
requests.post(f'{funcurl}?name=TSLA,FB&Train=True&code={funckey}==',timeout=50)
exit()
token=os.environ.get('IEXTestKey')
sybl='TWTR'
base_url = f'https://sandbox.iexapis.com/stable/stock/{sybl}/quote'
params={'token': token}
resp=requests.get(base_url,params=params)
d=resp.json()

print(d)
exit()

EST=pytz.timezone('US/Eastern')
print(datetime.now(EST).strftime('%Y-%m-%d'))

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