# -*- coding: utf-8 -*-
"""
Created on Fri May  6 12:19:29 2022

@author: ordersofmagnitude
"""

#

import pandas as pd
import pandas_datareader.data as web
from datetime import datetime
import requests
import json


def stock_getter(year, month, day, symbol):

    start = datetime(year, month, day)
    end = datetime.now()
    
    
    df = web.DataReader(symbol, "yahoo", start, end)
    df.to_csv(f"datasets/{symbol}.csv")
    #df = pd.read_csv(f"datasets/{symbol}.csv")


def get_bitcoin():
    url = 'https://www.alphavantage.co/query?function=DIGITAL_CURRENCY_WEEKLY&symbol=BTC&market=USD&apikey=90892ZD10G9MKC8E'
    r = requests.get(url)
    bitcoin = r.json()
    return bitcoin

#print(bitcoin)

#bitcoin.to_csv("dataset/bitcoin.csv")


def get_ethereum():

    url = 'https://www.alphavantage.co/query?function=DIGITAL_CURRENCY_DAILY&symbol=ETH&market=USD&apikey=90892ZD10G9MKC8E'
    r = requests.get(url)
    ethereum = r.json()
    ethereum = pd.DataFrame(ethereum)
    ethereum.to_csv("datasets/ethereum.csv")

def parser(func):
    get_cryptos = func()
    crypto = get_cryptos["Time Series (Digital Currency Daily)"]
    
    data_dict = {"open": [], "high": [], "low": [],
                    "close": [], "volume": [], "market_cap": []}
    
    for price_dict in list(crypto.values()):
        data_dict["open"].append(float(price_dict["1a. open (USD)"]))
        data_dict["high"].append(float(price_dict["2a. high (USD)"]))
        data_dict["low"].append(float(price_dict["3a. low (USD)"]))
        data_dict["close"].append(float(price_dict["4a. close (USD)"]))
        data_dict["volume"].append(float(price_dict["5. volume"]))
        data_dict["market_cap"].append(float(price_dict["6. market cap (USD)"]))
    
    data_df = pd.DataFrame(data_dict)
    data_df["date"] = crypto.keys()
    
    return data_df


def crypto_preprocessor(df):
    
    data_dict = {"open": [], "high": [], "low": [],
                    "close": [], "volume": [], "market_cap": [], "date": []}
    
    for i in range(7, len(df)):
        price_str = df["Time Series (Digital Currency Daily)"].iloc[i]
        price_str = price_str.replace(r"'", r'"')
        price_dict = json.loads(price_str)
        
        data_dict["open"].append(float(price_dict["1a. open (USD)"]))
        data_dict["high"].append(float(price_dict["2a. high (USD)"]))
        data_dict["low"].append(float(price_dict["3a. low (USD)"]))
        data_dict["close"].append(float(price_dict["4a. close (USD)"]))
        data_dict["volume"].append(float(price_dict["5. volume"]))
        data_dict["market_cap"].append(float(price_dict["6. market cap (USD)"]))
        data_dict["date"].append(df["Unnamed: 0"].iloc[i])
        
    
    data_dict = pd.DataFrame(data_dict)
    
    data_dict["date"] = data_dict["date"].astype("datetime64[ns]")
        
    return data_dict

#eth = crypto_preprocessor(ethereum)
#eth.info()
#eth.to_csv("datasets/ethereum.csv")