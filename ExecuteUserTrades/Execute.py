import logging

import azure.functions as func
import pyodbc
import pandas as pd
import os
from dotenv import load_dotenv
load_dotenv()
import alpaca_trade_api as tradeapi
import requests

def runtrades(key):
    conn_str='DRIVER={ODBC Driver 17 for SQL Server};SERVER='+os.environ.get('server')+ \
        ';DATABASE='+os.environ.get('database')+ \
            ';UID='+os.environ.get('dbusername')+ \
                ';PWD='+ os.environ.get('dbpassword')
    cnxn = pyodbc.connect(conn_str)       
    cursor = cnxn.cursor()

    cursor.execute(f"Select UserSecret,UserWatchlist,Live_Paper From TradeServiceUsers Where UserKey='{key}'")
    user = cursor.fetchone()
    passkey='test' if user[2]==0 else 'prod'
    api = tradeapi.REST(key,user[0],
        'https://paper-api.alpaca.markets', api_version='v2')
    
    accountinfo= api.get_account()
    acct_positions = api.list_positions()
    acct_totalbalance = float(accountinfo.portfolio_value)
    acct_availabletrade = float(accountinfo.cash)

    openpositions= [[p.symbol,p.qty] for p in acct_positions]

    cursor.execute("Select Symbol,Predicted_Inc, Model_Accuracy From Tickers Where trained_filename is not NULL and Model_Accuracy>0.7")
    predictions = pd.DataFrame([list(ele) for ele in cursor],columns=['Symbol','PredictedGrowth','Accuracy'])
    predictions.set_index('Symbol',inplace=True)
    predictions.sort_values(by='PredictedGrowth',ascending=False, inplace=True)
    #Check for equities to sell
    for ticker in openpositions:
        #trigger sell if the predicted growth goes below 0
        try:
            if predictions.at[ticker[0],'PredictedGrowth']<=0 or predictions.at[ticker[0],'PredictedGrowth']==None:
                api.submit_order(symbol=ticker[0],qty=ticker[1],side="sell")
        except:
            api.submit_order(symbol=ticker[0],qty=ticker[1],side="sell")

    #if available balance is <10% of total porfolio balance, trigger trades
    token=os.environ.get('IEXProdKey')
    params={'token': token}
    buycount=1
    if acct_totalbalance*0.1<acct_availabletrade:
        for pred in predictions.itertuples():
            portfolio = [i[0] for i in openpositions] if len(openpositions)!=0 else ['empty']
            if pred.PredictedGrowth>0 and not pred.Index in portfolio:
                symbol=pred.Index
                print(f'Buying: {symbol}')
                base_url = f'https://cloud.iexapis.com/stable/stock/{symbol}/quote'
                resp=requests.get(base_url,params=params)
                d=resp.json()

                api.submit_order(symbol=pred.Index,qty=round(acct_totalbalance*0.1/d["latestPrice"],0),side="buy",type = "limit",limit_price=d["latestPrice"])
                buycount+=1
                if buycount>2:
                    return

def main(req: func.HttpRequest) -> None:
    userkey = req.params.get('name')
    if userkey ==None:
        return
    else:
        runtrades(userkey)