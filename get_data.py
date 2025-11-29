import os
from pathlib import Path

vn_dir = Path.home() / ".vnstock"
os.makedirs(vn_dir, exist_ok=True)

from get_data import get_data
from clean_data import clean_data
from train_model import train_model
import datetime as dt
VN30 = [
    "ACB", "BID", "CTG", "FPT", "GVR", "HDB", "HPG", "MBB", "MSN",
    "MWG", "NVL", "PDR", "PLX", "POW", "SAB", "SSI", "STB", "TCB",
    "TPB", "VCB", "VHM", "VIC", "VJC", "VNM", "VPB", "VRE", "GAS", "KDH", "PNJ", "REE"
]
for symbol in VN30:
    start_date = '2015-01-01'
    end_date = dt.date.today().strftime('%Y-%m-%d')
    get_df = get_data(symbol, start_date, '2024-12-31')
    clean_df = clean_data(get_df, symbol)
    model = train_model(clean_df, symbol = symbol)
    print(f"Model saved to {model}")    
from vnstock import Vnstock
import pandas as pd

def get_data(ticker ,start_date_str, end_date_str):
    stock_symbol = Vnstock().stock(symbol = ticker, source = 'VCI')
    price_df = stock_symbol.quote.history(start = start_date_str,end = end_date_str)
    price_df = price_df.rename(columns = {'close':'price'}, inplace = False)
    price_df.reset_index(inplace = True)
    price_df = price_df[['time', 'price']]
    return price_df
