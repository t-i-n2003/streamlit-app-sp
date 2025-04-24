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