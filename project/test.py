import yfinance as yf
import numpy as np
import pandas as pd

# 애플 주식 데이터 다운로드
ticker = 'AAPL'
data = yf.download(ticker, period='1y', interval='1d')

# 종가만 추출
closing_prices = data['Close'].values

# 데이터 전처리: 예를 들어, 이전 종가를 예측하는 경우
X = []
y = []

# 타임스텝 설정
time_steps = 5  # 예를 들어, 5일의 데이터를 사용하여 다음 날을 예측

for i in range(len(closing_prices) - time_steps):
    X.append(closing_prices[i:i + time_steps])
    y.append(closing_prices[i + time_steps])

X = np.array(X)
y = np.array(y)

# 데이터 형태 변환
X = X.reshape((X.shape[0], X.shape[1], 1))  # (샘플 수, 타임스텝, 특성 수)



import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

class OneDCNN(Model):
    def __init__(self, input_shape, num_classes=1):
        super(OneDCNN, self).__init__()
        self.conv1 = Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=input_shape)
        self.pool = MaxPooling1D(pool_size=2)
        self.flatten = Flatten()
        self.dense1 = Dense(64, activation='relu')
        self.dense2 = Dense(num_classes, activation='linear')  # 회귀 문제이므로 linear 활성화 사용

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.dense1(x)
        return self.dense2(x)

    def train(self, X, y, epochs=5, batch_size=32):
        self.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])  # 평균 절대 오차

        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        # model_checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_loss')

        # 모델 학습 (validation_split을 통해 검증 데이터 설정)
        self.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=0.2, callbacks=[early_stopping])

# 모델 인스턴스 생성
input_shape = X.shape[1:]  # X의 형태에서 타임스텝과 특성 수 추출
model = OneDCNN(input_shape)

# 모델 학습
model.train(X, y, epochs=5)

# 모델 평가
loss, mae = model.evaluate(X, y)
print(f'Loss: {loss}, MAE: {mae}')
