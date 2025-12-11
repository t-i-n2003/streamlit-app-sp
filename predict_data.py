from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
import joblib

def predict(df, symbol, model_path, num_days):
    model = load_model(model_path)    
    struc_df = pd.read_csv('best_model_results.csv')
    timestep = struc_df[struc_df['symbol'] == symbol]['best_time_steps'].values[0]

    # lấy input cuối
    input_data = df['price'].iloc[-timestep:].values.reshape(-1, 1)

    scaler = joblib.load(f'Scalers/{symbol}_scaler.pkl')
    scaled_input = scaler.transform(input_data)

    predictions = []

    for i in range(num_days):
        X = scaled_input.reshape(1, timestep, 1)

        # model → scaled value
        pred_scaled = float(model.predict(X, verbose=0)[0][0])

        # scaled → actual price
        pred_real = scaler.inverse_transform([[pred_scaled]])[0][0]

        predictions.append(pred_real)

        # convert actual price → scaled lại để đưa vào mô hình
        new_scaled = scaler.transform([[pred_real]])

        # update input window
        scaled_input = np.vstack([scaled_input[1:], new_scaled])

    # KHÔNG inverse_transform lần nữa!
    predictions = pd.DataFrame(predictions, columns=['predicted_price'])

    next_date = df['time'].iloc[-1] + pd.DateOffset(days=1)
    future_dates = pd.bdate_range(start=next_date, periods=num_days, freq='B')

    df_predictions = pd.DataFrame({
        'time': future_dates,
        'predicted_price': predictions['predicted_price']
    })

    return df_predictions
