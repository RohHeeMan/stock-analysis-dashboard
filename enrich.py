<<<<<<< HEAD
# enrich.py — EPS, BPS, PER, PBR 계산 (DB에서 주가/주식수 조회)
import pandas as pd
from sqlalchemy import text


def enrich_with_valuation_ratios(df_summary: pd.DataFrame, engine) -> pd.DataFrame:
    df = df_summary.copy()
    df['period'] = df['fiscal_year'].astype(str) + df['fiscal_qtr']

    # DB에서 시장 데이터 로드
    sql = text("""
        SELECT p.stock_code, TO_CHAR(p.date, 'YYYY') ||
               CASE
                 WHEN EXTRACT(MONTH FROM p.date)::INT <= 3 THEN '1Q'
                 WHEN EXTRACT(MONTH FROM p.date)::INT <= 6 THEN '2Q'
                 WHEN EXTRACT(MONTH FROM p.date)::INT <= 9 THEN '3Q'
                 ELSE 'FY'
               END AS period,
               p.close_price, s.shares_outstanding
        FROM stock_prices p
        JOIN stock_shares s ON p.stock_code = s.stock_code AND p.date = s.date
    """)
    df_market = pd.read_sql(sql, engine)
    df_market.rename(columns={"stock_code": "ticker"}, inplace=True)

    # 문자열 처리
    df['ticker'] = df['ticker'].astype(str)
    df['period'] = df['period'].astype(str)
    df_market['ticker'] = df_market['ticker'].astype(str)
    df_market['period'] = df_market['period'].astype(str)

    # 병합
    df = pd.merge(df, df_market, on=['ticker', 'period'], how='left')

    # 지표 계산
    df['eps'] = df['net_income'] / df['shares_outstanding']
    df['bps'] = df['equity'] / df['shares_outstanding']
    df['per'] = df['close_price'] / df['eps']
    df['pbr'] = df['close_price'] / df['bps']

    df[['eps', 'bps', 'per', 'pbr']] = df[['eps', 'bps', 'per', 'pbr']].round(2)

    # 컬럼명 통일
    df.rename(columns={
        'close_price': 'stock_price'
    }, inplace=True)

    return df
=======
# enrich.py — EPS, BPS, PER, PBR 계산 (DB에서 주가/주식수 조회)
import pandas as pd
from sqlalchemy import text


def enrich_with_valuation_ratios(df_summary: pd.DataFrame, engine) -> pd.DataFrame:
    df = df_summary.copy()
    df['period'] = df['fiscal_year'].astype(str) + df['fiscal_qtr']

    # DB에서 시장 데이터 로드
    sql = text("""
        SELECT p.stock_code, TO_CHAR(p.date, 'YYYY') ||
               CASE
                 WHEN EXTRACT(MONTH FROM p.date)::INT <= 3 THEN '1Q'
                 WHEN EXTRACT(MONTH FROM p.date)::INT <= 6 THEN '2Q'
                 WHEN EXTRACT(MONTH FROM p.date)::INT <= 9 THEN '3Q'
                 ELSE 'FY'
               END AS period,
               p.close_price, s.shares_outstanding
        FROM stock_prices p
        JOIN stock_shares s ON p.stock_code = s.stock_code AND p.date = s.date
    """)
    df_market = pd.read_sql(sql, engine)
    df_market.rename(columns={"stock_code": "ticker"}, inplace=True)

    # 문자열 처리
    df['ticker'] = df['ticker'].astype(str)
    df['period'] = df['period'].astype(str)
    df_market['ticker'] = df_market['ticker'].astype(str)
    df_market['period'] = df_market['period'].astype(str)

    # 병합
    df = pd.merge(df, df_market, on=['ticker', 'period'], how='left')

    # 지표 계산
    df['eps'] = df['net_income'] / df['shares_outstanding']
    df['bps'] = df['equity'] / df['shares_outstanding']
    df['per'] = df['close_price'] / df['eps']
    df['pbr'] = df['close_price'] / df['bps']

    df[['eps', 'bps', 'per', 'pbr']] = df[['eps', 'bps', 'per', 'pbr']].round(2)

    # 컬럼명 통일
    df.rename(columns={
        'close_price': 'stock_price'
    }, inplace=True)

    return df
>>>>>>> 25399d5885ee61e9b7704a7250bffee00d20070d
