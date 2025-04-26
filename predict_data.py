from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
import joblib

def predict(df, symbol, model_path, num_days):
    model = load_model(model_path)    
    struc_df = pd.read_csv('best_model_results.csv')
    timestep = struc_df[struc_df['symbol'] == symbol]['best_time_steps'].values[0]
    input_data = df['price'].iloc[-timestep:].values.reshape(-1, 1)
    scaler = joblib.load(f'Scalers/{symbol}_scaler.pkl')
    scaled_input = scaler.fit_transform(input_data)  # ✅ chỉ transform, KHÔNG fit_transform
    predictions = []
    for i in range(num_days):
        input_reshaped = scaled_input.reshape(1, timestep, 1)
        prediction = model.predict(input_reshaped, verbose=0)
        predictions.append(prediction[0][0])
        
        new_input = np.array([[prediction[0][0]]])
        scaled_input = np.vstack([scaled_input[1:], new_input])  # giữ shape (timestep, 1)
    
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    predictions = pd.DataFrame(predictions, columns=['predictions'])

    next_date = df['time'].iloc[-1] + pd.DateOffset(days=1)
    future_dates = pd.bdate_range(start=next_date, periods=num_days, freq='B')

    df_predictions = pd.DataFrame({
        'time': future_dates,
        'predicted_price': predictions['predictions']
    })
    
    return df_predictions
