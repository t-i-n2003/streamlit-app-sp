import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.losses import MeanSquaredLogarithmicError
from scikeras.wrappers import KerasRegressor
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import EarlyStopping
import os
def create_lstm_model(units=50, dropout_rate=0.2, time_steps=10):
    model = Sequential()
    model.add(LSTM(units=units, activation='relu', input_shape=(time_steps, 1)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss=MeanSquaredLogarithmicError())
    return model

def mean_bias_error(y_true, y_pred):
    return K.mean(y_pred - y_true)

def train_model(df, symbol):
    param_grid = {
        'model__units': [50],
        'model__dropout_rate': [0.2, 0.15],
        'epochs': [20],
        'batch_size': [32, 64],
    }
    time_steps_list = [10, 20]

    def create_and_split_data(df, time_steps):
        X_seq, y_seq = [], []
        for i in range(len(df) - time_steps):
            X_seq.append(df['price'].iloc[i:i + time_steps])
            y_seq.append(df['price'].iloc[i + time_steps])
        X_seq = np.array(X_seq).reshape(-1, time_steps, 1)
        y_seq = np.array(y_seq)
        return train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

    best_result = None

    for ts in time_steps_list:
        X_train, X_test, y_train, y_test = create_and_split_data(df, ts)

        model = KerasRegressor(
            model=create_lstm_model,
            model__time_steps=ts,
            verbose=0
        )

        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_grid,
            n_iter=5,
            cv=2,
            scoring='neg_mean_squared_error',
            verbose=1
        )
        search.fit(X_train, y_train)

        if best_result is None or search.best_score_ > best_result.best_score_:
            best_result = search
            best_X_test, best_y_test = X_test, y_test

    best_params = best_result.best_params_
    best_time_steps = best_result.estimator.model__time_steps

    # Create best model again for full training
    best_model = create_lstm_model(
        units=best_params['model__units'],
        dropout_rate=best_params['model__dropout_rate'],
        time_steps=best_time_steps
    )
    X_train_best, X_test_best, y_train_best, y_test_best = create_and_split_data(df, best_time_steps)

    early_stop = EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)

    best_model.fit(
        X_train_best,
        y_train_best,
        epochs=best_params['epochs'],
        batch_size=best_params['batch_size'],
        callbacks=[early_stop],
        verbose=0
    )

    best_model_path = f'Models/best_{symbol}_rdsearch_model.h5'
    best_model.save(best_model_path)

    y_pred = best_model.predict(X_test_best)
    mse = mean_squared_error(y_test_best, y_pred)
    mae = mean_absolute_error(y_test_best, y_pred)
    mbe = K.eval(mean_bias_error(K.constant(y_test_best), K.constant(y_pred.flatten())))
    msle = K.eval(MeanSquaredLogarithmicError()(K.constant(y_test_best), K.constant(y_pred.flatten())))

    results_df = pd.DataFrame({
        'symbol': [symbol],
        'best_hyperparameters': [best_params],
        'best_time_steps': [best_time_steps],
        'mse': [mse],
        'mae': [mae],
        'mbe': [mbe],
        'msle': [msle],
        'best_model_path': [best_model_path]
    })
    save_path = 'best_model_results.csv'
    try:
        if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
            existing_df = pd.read_csv(save_path)
            updated_df = pd.concat([existing_df, results_df], ignore_index=True)
            updated_df.to_csv(save_path, index=False)
        else:
            results_df.to_csv(save_path, index=False)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        results_df.to_csv(save_path, index=False)

    print(f"✅ Best score: {best_result.best_score_}")
    print(f"✅ Best params: {best_params}")
    return best_model_path
