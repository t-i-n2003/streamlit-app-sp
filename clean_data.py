from sklearn.preprocessing import MinMaxScaler
import joblib
def clean_data(df, symbol):
    df = df.drop_duplicates()
    scaler = MinMaxScaler() 
    df['price'] = scaler.fit_transform(df[['price']])
    joblib.dump(scaler, f'Scalers/{symbol}_scaler.pkl')
    return df