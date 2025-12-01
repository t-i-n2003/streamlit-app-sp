import pandas as pd
from vnstock import stock_historical_data

def get_data(ticker, start_date_str, end_date_str):
    # Lấy dữ liệu giá từ API cũ của vnstock 0.2.3
    df = stock_historical_data(
        symbol=ticker,
        start_date=start_date_str,
        end_date=end_date_str,
        resolution='1D'
    )

    # Chuẩn hóa cột cho giống code cũ
    df = df.rename(columns={'close': 'price'})
    df = df[['time', 'price']]

    # Reset index nếu cần
    df = df.reset_index(drop=True)

    return df
