import os
from pathlib import Path

vn_dir = Path.home() / ".vnstock"
os.makedirs(vn_dir, exist_ok=True)    
from vnstock import Vnstock
import pandas as pd

def get_data(ticker ,start_date_str, end_date_str):
    stock_symbol = Vnstock().stock(symbol = ticker, source = 'VCI')
    price_df = stock_symbol.quote.history(start = start_date_str,end = end_date_str)
    price_df = price_df.rename(columns = {'close':'price'}, inplace = False)
    price_df.reset_index(inplace = True)
    price_df = price_df[['time', 'price']]
    return price_df
