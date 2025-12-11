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

    predictions_scaled = []   # ❗ lưu giá scaled, chưa inverse

    for i in range(num_days):
        X = scaled_input.reshape(1, timestep, 1)

        # model xuất ra giá scaled
        pred_scaled = float(model.predict(X, verbose=0)[0][0])

        # lưu giá scaled
        predictions_scaled.append(pred_scaled)

        # update input window (scaled → giữ nguyên)
        scaled_input = np.vstack([scaled_input[1:], [[pred_scaled]]])

    # ❗ Chỉ inverse_transform 1 lần, sau khi loop kết thúc
    predictions_real = scaler.inverse_transform(
        np.array(predictions_scaled).reshape(-1,1)
    )

    df_predictions = pd.DataFrame({
        "time": pd.bdate_range(
            start=df['time'].iloc[-1] + pd.DateOffset(days=1),
            periods=num_days,
            freq='B'
        ),
        "predicted_price": predictions_real.flatten()
    })

    return df_predictions
