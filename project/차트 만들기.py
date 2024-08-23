import pandas as pd
import pymysql
import plotly.graph_objects as go
import DBconnection as db

# 데이터베이스 연결
connection = db.connection()

# 원하는 종목 코드, 날짜 입력
print('종목 코드 입력')
code = input()
print('날짜 입력(형식 YYYY-MM-DD)')
date = input()

# SQL 쿼리 실행
query = db.select_sql(code=code, date=date)

# 데이터프레임으로 변환
df = pd.read_sql(query, connection)

# 데이터베이스 연결 종료
connection.close()

# 날짜를 인덱스로 설정
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

# Plotly를 사용하여 캔들스틱 차트 그리기
fig = go.Figure(data=[go.Candlestick(x=df.index,
                                       open=df['open_price'],
                                       high=df['high_price'],
                                       low=df['low_price'],
                                       close=df['close_price'],
                                       name='Candlestick'),
                      go.Scatter(x=df.index, y=df['close_price'], mode='lines', name='Close Price', line=dict(color='blue'))])

# 레이아웃 설정
fig.update_layout(
    title='Stock Price Chart for 2024-08-12',
    xaxis_title='Date',
    yaxis_title='Price',
    xaxis_rangeslider_visible=False,
    hovermode='x unified'
)

# 확대/축소 기능을 위한 설정
fig.update_xaxes(
    rangeslider_visible=False,  # 범위 슬라이더 비활성화
    showgrid=True,              # 그리드 표시
    autorange=True              # 자동 범위 설정
)

fig.update_yaxes(
    showgrid=True               # 그리드 표시
)

# 차트 표시
fig.show()

