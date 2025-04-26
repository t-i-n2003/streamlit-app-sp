# cài thư viện + chạy câu lệnh ở terminal streamlit run .\streamlit_stock.py
#pip install pandas streamlit numpy==1.26.4 plotly pandas_ta vnstock
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import predict_data
import get_data as gd
import datetime as dt
from tenacity import RetryError
import time
st.set_page_config(page_title="Stock Prediction", layout="wide")

VN30 = [
    "ACB", "BID", "CTG", "FPT", "GVR", "HDB", "HPG", "MBB", "MSN",
    "MWG", "NVL", "PDR", "PLX", "POW", "SAB", "SSI", "STB", "TCB",
    "TPB", "VCB", "VHM", "VIC", "VJC", "VNM", "VPB", "VRE", "GAS", "KDH", "PNJ", "REE"
]

def plot(df, symbol):
    fig = make_subplots(rows=1, cols=1, vertical_spacing=0.1, subplot_titles=("Giá cổ phiếu dư đoán",))
    fig.add_trace(go.Scatter(x=df['time'], y=df['Predicted Price'], mode='lines', name='Giá dự đoán'), row=1, col=1)
    fig.update_layout(title_text=f"Biểu đồ giá dự đoán cổ phiếu {symbol }", height=600)
    return fig

st.title("Dữ liệu chứng khoán")

tabs = st.tabs(["Dự đoán giá cổ phiếu", "kiểm tra hiệu suất sinh lời"])
# ============================ 🟢 TAB 1: Dữ liệu lịch sử giao dịch ============================
with tabs[0]:
    st.header("Nhập thông tin dữ liệu chứng khoán")
    col1, col2 = st.columns(2)
    with col1:
        symbol = st.selectbox("Chọn mã cổ phiếu:", VN30, index=0)
        st.write("### Thời gian dự đoán")
        start_date = st.date_input("Ngày bắt đầu dự đoán:", value= pd.to_datetime('2025-01-01'), max_value= dt.date.today())
        min = pd.to_datetime(start_date).strftime('%Y-%m-%d')
    with col2:
        numbers_date = st.number_input("Nhập số ngày cần dự đoán: ", step = 1)
    if st.button("Dự đoán"):
        model_path = f'Models/best_{symbol}_rdsearch_model.h5'
        st.write("### Giá dự đoán")
        min_date = pd.to_datetime(min)
        fromdate = min_date - pd.DateOffset(days=100)
        fromdate = pd.to_datetime(fromdate).strftime('%Y-%m-%d')
        price_df = gd.get_data(symbol, fromdate, min)   
        price_df.drop_duplicates(subset=['time'], inplace=True)
        df = predict_data.predict(price_df, symbol, model_path, numbers_date)
        st.dataframe(df)
        st.write("### Biểu đồ dự đoán")
        st.plotly_chart(plot(df, symbol))   
            
with tabs[1]:
    st.header("Nhập thông tin dữ liệu chứng khoán")
    tab11, tab12 = st.columns(2)
    with tab11:
        symbol1 = st.selectbox("Chọn mã cổ phiếu: ", VN30, index=0, key=1)
        st.write("### Thời gian kiểm tra hiệu suất sinh lời")
        start_date1 = st.date_input("Ngày bắt đầu dự đoán kiểm tra: ", value= pd.to_datetime('2025-01-01'), max_value= dt.date.today() - pd.DateOffset(days=15))
        end_Date1 = st.date_input("Ngày kết thúc dự đoán kiểm tra: ", value= pd.to_datetime('2025-01-02'), max_value= dt.date.today())
        min1 = pd.to_datetime(start_date).strftime('%Y-%m-%d')
    with tab12:
        profit = st.number_input("Hiệu suất sinh lời: ", step=0.001, format="%.3f")
        cut_loss = st.number_input("Giá trị cắt lỗ: ", step=0.001, format="%.3f")
        real_df = pd.DataFrame(columns=[
                "date", "symbol", "price", "volume", "sell", "buy", "total_value", "holding_volume"
            ])
    if st.button("Dự đoán", key=2):
        model_path = f'Models/best_{symbol1}_rdsearch_model.h5'
        total_value = 0  # tổng giá trị lời/lỗ

        current_date = pd.to_datetime(start_date1)
        end_Date = pd.to_datetime(end_Date1)
        total_value = 0
        cash = 50000        
        holding_volume = 0
        while current_date <= end_Date:
            if current_date.weekday() in [4, 5]:
                current_date += pd.DateOffset(days=1)
                continue
            from_date = current_date - pd.DateOffset(days=100)
            from_date_str = from_date.strftime('%Y-%m-%d')
            current_date_str = current_date.strftime('%Y-%m-%d')

            # Dữ liệu đầu vào
            price_df = gd.get_data(symbol1, from_date_str, current_date_str)
            time.sleep(0.5)
            price_df.drop_duplicates(subset=['time'], inplace=True)

            if price_df.empty:
                current_date += pd.DateOffset(days=1)
                continue

            past_price = price_df['price'].iloc[-1]
            df_pred = predict_data.predict(price_df, symbol1, model_path, num_days=1)

            if df_pred is None or df_pred.empty:
                current_date += pd.DateOffset(days=1)
                continue

            predicted_price = df_pred['predicted_price'].iloc[-1]
            predict_date = pd.to_datetime(df_pred['time'].iloc[-1])

            # Giá thực tế hôm nay
            try:
                actual_df = gd.get_data(symbol1, current_date_str, current_date_str)
                time.sleep(0.5)
                actual_price = actual_df['price'].iloc[0]
                actual_df.drop_duplicates(subset=['time'], inplace=True)
            except (ValueError, RetryError) as e:
                current_date += pd.DateOffset(days=1)
                continue
            if actual_df.empty:
                current_date += pd.DateOffset(days=1)
                continue
            trade = None

            # Điều kiện mua
            if predicted_price >= past_price * (1 + profit):
                if cash < 50 * past_price:
                    current_date += pd.DateOffset(days=1)
                    continue
                volume = 50
                holding_volume += volume
                cash -= volume * past_price  # trừ tiền mua

                trade = {
                    "date": predict_date,
                    "symbol": symbol1,
                    "price": past_price,
                    "volume": volume,
                    "sell": 0,
                    "buy": 1,
                    "total_value": cash + holding_volume * past_price,
                    'holding_volume': holding_volume
                }

            # Điều kiện bán
            elif predicted_price <= past_price * (1 - cut_loss):
                if holding_volume < 0:
                    current_date += pd.DateOffset(days=1)
                    continue
                volume = 50
                holding_volume -= volume
                cash += volume * past_price  # cộng tiền bán

                trade = {
                    "date": predict_date,
                    "symbol": symbol1,
                    "price": past_price,
                    "volume": -volume,
                    "sell": 1,
                    "buy": 0,
                    "total_value": cash + holding_volume * past_price,
                    'holding_volume': holding_volume
                }
            if trade:
                real_df = pd.concat([real_df, pd.DataFrame([trade])], ignore_index=True)

            current_date += pd.DateOffset(days=1)

        st.dataframe(real_df)
