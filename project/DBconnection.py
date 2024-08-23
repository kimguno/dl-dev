import pymysql

def connection():
    connection = pymysql.connect(
        host='192.168.40.53',       # 데이터베이스 호스트
        user='root',                # 사용자 이름
        password='big185678',       # 비밀번호
        database='finaldb',         # 데이터베이스 이름
    )
    return connection

def select_sql(code, date):
    query = f"""
            SELECT am.code,
                am.open_price,
                am.high_price,
                am.low_price,
                am.close_price,
                am.volume,
                STR_TO_DATE(CONCAT(DATE(Jdate), ' ', 
                LPAD(FLOOR(time / 100), 2, '0'), ':', 
                LPAD(MOD(time, 100), 2, '0')), '%Y-%m-%d %H:%i') as date
            FROM 
                A{code}_mindata am
            WHERE 
                DATE(Jdate) = '{date}';
            """
    return query

def execute_query(code, date):
    conn = connection()
    if conn is None:
        return None
    
    try:
        query = select_sql(code, date)
        with conn.cursor() as cursor:
            cursor.execute(query)
            result = cursor.fetchall()  # 모든 결과를 가져옵니다.
            return result
    except pymysql.MySQLError as e:
        print(f"Error executing query: {e}")
        return None
    finally:
        conn.close()  # 커넥션 종료

