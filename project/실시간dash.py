import dash
from dash import dcc, html
import pandas as pd
import pymysql
import plotly.graph_objects as go
import DBconnection as db
import time

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Real-Time Stock Price Chart"),
    dcc.Graph(id='live-graph'),
    dcc.Interval(id='interval-component', interval=10*1000, n_intervals=0)  # 10초마다 업데이트
])
@app.callback(
    dash.dependencies.Output('live-graph', 'figure'),
    [dash.dependencies.Input('interval-component', 'n_intervals')]
)
def update_graph(n):
    try:
        connection = db.connection()
        if connection is None:
            print("데이터베이스 연결 실패")
            return go.Figure()  # 빈 차트 반환
        
        code = 'A000660'  # 예시 종목 코드
        date = time.strftime("%Y-%m-%d")  # 현재 날짜
        
        query = db.select_sql(code=code, date=date)
        df = pd.read_sql(query, connection)
        connection.close()

        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)

        fig = go.Figure(data=[go.Candlestick(x=df.index,
                                               open=df['open_price'],
                                               high=df['high_price'],
                                               low=df['low_price'],
                                               close=df['close_price'],
                                               name='Candlestick'),
                              go.Scatter(x=df.index, y=df['close_price'], mode='lines', name='Close Price', line=dict(color='blue'))])
        
        fig.update_layout(title='Stock Price Chart', xaxis_title='Date', yaxis_title='Price', xaxis_rangeslider_visible=False, hovermode='x unified')
        
        return fig
    
    except Exception as e:
        print(f"오류 발생: {e}")
        return go.Figure()  # 빈 차트 반환

if __name__ == '__main__':
    app.run_server(debug=True)