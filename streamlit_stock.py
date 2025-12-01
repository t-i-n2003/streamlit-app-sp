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
from NLP import predict_label
st.set_page_config(page_title="Stock Prediction", layout="wide")

VN30 = [
    "ACB", "BID", "CTG", "FPT", "GVR", "HDB", "HPG", "MBB", "MSN",
    "MWG", "NVL", "PDR", "PLX", "POW", "SAB", "SSI", "STB", "TCB",
    "TPB", "VCB", "VHM", "VIC", "VJC", "VNM", "VPB", "VRE", "GAS", "KDH", "PNJ", "REE"
]
def plot(df, symbol):
    df = df.copy()
    df = df.sort_values('time')

    # convert predicted price
    df['predicted_price'] = (
        df['predicted_price']
        .astype(str)
        .str.replace(',', '')
        .str.replace(' ', '')
        .astype(float)
    )

    # convert time to datetime
    df['time'] = pd.to_datetime(df['time'])

    fig = make_subplots(rows=1, cols=1, vertical_spacing=0.1,
                        subplot_titles=("Giá cổ phiếu dự đoán",))

    fig.add_trace(
        go.Scatter(
            x=df['time'], 
            y=df['predicted_price'],
            mode='lines', 
            name='Giá dự đoán'
        ),
        row=1, col=1
    )

    fig.update_layout(
        title_text=f"Biểu đồ giá dự đoán cổ phiếu {symbol}",
        height=600
    )

    return fig

st.title("Dữ liệu chứng khoán")

tabs = st.tabs(["Dự đoán giá cổ phiếu", "Tác động tin tức tài chính"])
# ============================ 🟢 TAB 1: Dữ liệu lịch sử giao dịch ============================
with tabs[0]:
    st.header("Nhập thông tin dữ liệu chứng khoán")
    col1, col2 = st.columns(2)
    with col1:
        symbol = st.selectbox("Chọn mã cổ phiếu:", VN30, index=0)
        st.write("### Thời gian dự đoán")
        start_date = st.date_input("Ngày bắt đầu dự đoán:", value= pd.to_datetime('2024-06-02'), max_value= dt.date.today())
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
        st.write(df.head())
        st.write(df.dtypes)
        st.write(df.shape)
        st.write(df.isna().sum())
        df['predicted_price'] = df['predicted_price'].astype('float64')
        st.write(df.dtypes)
        st.dataframe(df)   
        st.write("### Biểu đồ dự đoán")
        st.plotly_chart(plot(df, symbol)) 
with tabs[1]:
    label_map = {
        0: "Tiêu cực",
        1: "Trung tính",
        2: "Tích cực"
    }

    text = st.text_input("Nhập tin tức: ", value="", key=2)

    if st.button("Dự đoán nhãn tin tức", key=3):
        if text:
            label_index, probs = predict_label(text)
            label_name = label_map[label_index]
            st.markdown(f"**Tin tức:** {text}")
            st.markdown(f"**Nhãn dự đoán:** {label_name}")
        else:
            st.warning("Vui lòng nhập tin tức để dự đoán nhãn.")

    
            
    
