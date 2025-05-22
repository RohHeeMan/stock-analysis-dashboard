import io
import pandas as pd
from datetime import datetime, timedelta
from sqlalchemy import text
from sqlalchemy.engine import Engine
from pykrx import stock

import logging

# 경고 숨기기
pd.set_option('future.no_silent_downcasting', True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def adjust_to_trading_day(ds: str, prev=True):
    try:
        # 실제 영업일 찾기
        nearest = stock.get_nearest_business_day_in_a_week(ds, prev=prev)
        return nearest
    except Exception as e:
        logging.warning(f"조정 중 오류 발생 (날짜: {ds}): {e}")
        return None  # 혹은 fallback 날짜 리턴

def _fetch_quarter_api_data(year: int, qtr: str, df_codes: pd.DataFrame) -> pd.DataFrame:
    """
    분기별 KRX API 호출
    반환: ticker, shares_outstanding, market_price,
           eps, bps, per, pbr, actual_date,
           fiscal_year, fiscal_qtr
    """
    quarter_end = {'1Q':'0331','2Q':'0630','3Q':'0930','FY':'1231'}
    ds_raw = f"{year}{quarter_end[qtr]}"
    ds_tr  = adjust_to_trading_day(ds_raw)

    # 1) 발행주식수 조회
    cap = stock.get_market_cap_by_ticker(ds_tr, market='ALL')[['상장주식수']]
    cap.columns = ['shares_outstanding']
    valid = set(df_codes['stock_code'])
    cap = cap.loc[cap.index.intersection(valid)]

    # 2) EPS/BPS 조회 (30일 lookback)
    def get_eps_bps(tkr: str):
        start = (datetime.strptime(ds_tr, '%Y%m%d') - timedelta(days=30)).strftime('%Y%m%d')
        df_f   = stock.get_market_fundamental(start, ds_tr, tkr)
        df_pos = df_f[df_f['EPS'] > 0]
        if not df_pos.empty:
            first     = df_pos.index[0]
            ds_actual = adjust_to_trading_day(first.strftime('%Y%m%d'))
            eps_val   = float(df_pos.at[first, 'EPS'])
        else:
            ds_actual = ds_tr
            eps_val   = pd.NA
        df_end = stock.get_market_fundamental(ds_tr, ds_tr, tkr)
        bps_val = int(df_end['BPS'].iloc[-1]) if not df_end.empty else pd.NA
        return ds_actual, eps_val, bps_val

    eps_data = [get_eps_bps(t) for t in cap.index]
    eps_df   = pd.DataFrame(eps_data, index=cap.index, columns=['actual_date','eps','bps'])

    # 3) 시가 조회
    price_dict = {}
    for t in cap.index:
        df_price = stock.get_market_ohlcv_by_date(ds_tr, ds_tr, t)
        price_dict[t] = int(df_price['종가'].iloc[-1]) if not df_price.empty else pd.NA
    price_df = pd.DataFrame.from_dict(price_dict, orient='index', columns=['market_price'])

    # 4) 병합 및 비율 계산
    df = cap.join(eps_df).join(price_df)
    df['per'] = df['market_price'] / df['eps']
    df['pbr'] = df['market_price'] / df['bps']

    # 5) 소수점 처리
    df['eps'] = df['eps'].astype('Float64').round(2)
    df['bps'] = df['bps'].astype('Int64')
    df['per'] = df['per'].round(2)
    df['pbr'] = df['pbr'].round(2)
    df['market_price'] = df['market_price'].astype('Int64')
    df['shares_outstanding'] = df['shares_outstanding'].astype('Int64')

    df.index.name = 'ticker'
    df.reset_index(inplace=True)
    df['fiscal_year'] = year
    df['fiscal_qtr']  = qtr

    return df[['ticker','shares_outstanding','market_price','eps','bps','per','pbr','actual_date','fiscal_year','fiscal_qtr']]

def fetch_quarter_data(year: int, qtr: str, df_codes: pd.DataFrame, engine: Engine) -> pd.DataFrame:
    """
    캐시 조회 후 누락 종목 및 0값 종목 재조회,
    DART 재무제표로 EPS/BPS/PER 보정
    """
    sql = text("""
        SELECT ticker, shares_outstanding, market_price,
               eps, bps, per, pbr, actual_date,
               fiscal_year, fiscal_qtr
          FROM krx_cache
         WHERE fiscal_year = :year
           AND fiscal_qtr  = :qtr
    """)
    df_cached = pd.read_sql(sql, engine, params={'year': year, 'qtr': qtr})

    # 1) 캐시에 없는 종목
    existing = set(df_cached['ticker'])
    df_missing = df_codes.loc[~df_codes['stock_code'].isin(existing)]
    missing_set = set(df_missing['stock_code'])

    # 2) 0값 종목
    bad = set(df_cached.loc[(df_cached['eps'] <= 0) | (df_cached['bps'] <= 0), 'ticker'])

    # 3) 재조회 대상
    to_fetch = missing_set.union(bad)
    df_new_codes = df_codes.loc[df_codes['stock_code'].isin(to_fetch)]

    df_new = pd.DataFrame()
    if not df_new_codes.empty:
        # API 호출
        df_new = _fetch_quarter_api_data(year, qtr, df_new_codes)
        df_new['eps'] = df_new['eps'].astype('Float64')
        df_new['bps'] = df_new['bps'].astype('Int64')

        # DART 불러오기 및 결합
        tickers = df_new['ticker'].tolist()
        if not tickers:
            # tickers가 비어있으면 빈 DataFrame 반환(상장폐지된 경우)
            df_equity = pd.DataFrame(columns=['ticker', 'fiscal_year', 'fiscal_qtr', 'equity_parent'])
            df_profit = pd.DataFrame(columns=['ticker', 'fiscal_year', 'fiscal_qtr', 'profit_parent'])
        else:
            df_equity = pd.read_sql(
                text("""
                    SELECT ticker, fs_year AS fiscal_year,
                        fs_qtr  AS fiscal_qtr,
                        thstrm_amount::numeric AS equity_parent
                    FROM raw_financials
                    WHERE account_id = 'ifrs-full_EquityAttributableToOwnersOfParent'
                    AND fs_div     = 'CFS'
                    AND ticker     IN :tickers
                    AND fs_year    = :year
                    AND fs_qtr     = :qtr
                """),
                engine, params={'tickers': tuple(tickers), 'year': year, 'qtr': qtr}
            )
            df_profit = pd.read_sql(
                text("""
                    SELECT ticker, fs_year AS fiscal_year,
                        fs_qtr  AS fiscal_qtr,
                        thstrm_amount::numeric AS profit_parent
                    FROM raw_financials
                    WHERE account_id = 'ifrs-full_ProfitLossAttributableToOwnersOfParent'
                    AND fs_div = 'CFS'
                    AND ticker IN :tickers
                    AND fs_year    = :year
                    AND fs_qtr     = :qtr
                """),
                engine, params={'tickers': tuple(tickers), 'year': year, 'qtr': qtr}
            )        
        # df_equity = pd.read_sql(
        #     text("""
        #         SELECT ticker, fs_year AS fiscal_year,
        #                fs_qtr  AS fiscal_qtr,
        #                thstrm_amount::numeric AS equity_parent
        #           FROM raw_financials
        #          WHERE account_id = 'ifrs-full_EquityAttributableToOwnersOfParent'
        #            AND fs_div     = 'CFS'
        #            AND ticker     IN :tickers
        #            AND fs_year    = :year
        #            AND fs_qtr     = :qtr
        #     """),
        #     engine, params={'tickers': tuple(tickers), 'year': year, 'qtr': qtr}
        # )
        # df_profit = pd.read_sql(
        #     text("""
        #         SELECT ticker, fs_year AS fiscal_year,
        #                fs_qtr  AS fiscal_qtr,
        #                thstrm_amount::numeric AS profit_parent
        #           FROM raw_financials
        #          WHERE account_id = 'ifrs-full_ProfitLossAttributableToOwnersOfParent'
        #            AND fs_div = 'CFS'
        #            AND ticker IN :tickers
        #            AND fs_year    = :year
        #            AND fs_qtr     = :qtr
        #     """),
        #     engine, params={'tickers': tuple(tickers), 'year': year, 'qtr': qtr}
        # )

        df_new = df_new.merge(df_equity, on=['ticker', 'fiscal_year', 'fiscal_qtr'], how='left') \
                       .merge(df_profit, on=['ticker', 'fiscal_year', 'fiscal_qtr'], how='left')

        # EPS/BPS/PER 보정
        mask_bps = (df_new['bps'].isna() | (df_new['bps'] == 0)) & df_new['equity_parent'].notna()
        df_new.loc[mask_bps, 'bps'] = (df_new.loc[mask_bps, 'equity_parent'] / df_new.loc[mask_bps, 'shares_outstanding']).round(0).astype('Int64')

        mask_eps = (df_new['eps'].isna() | (df_new['eps'] == 0)) & df_new['profit_parent'].notna()
        df_new.loc[mask_eps, 'eps'] = (df_new.loc[mask_eps, 'profit_parent'] / df_new.loc[mask_eps, 'shares_outstanding']).round(2)

        mask_per = (df_new['per'].isna() | (df_new['per'] == 0)) & df_new['eps'].notna() & (df_new['eps'] != 0)
        df_new.loc[mask_per, 'per'] = (df_new.loc[mask_per, 'market_price'] / df_new.loc[mask_per, 'eps']).round(2)

        df_new.drop(columns=['equity_parent', 'profit_parent'], inplace=True)

        # 캐시 업데이트
        cache_quarter_to_db(df_new, engine)

        # 4) concat 경고 방지 및 컬럼 정리
        df_new.dropna(how="all", axis=1, inplace=True)

    valid_cached = df_cached.loc[(df_cached['eps'] > 0) & (df_cached['bps'] > 0)]
    
    # 1) df_new가 비어 있으면 valid_cached만 반환
    if df_new.empty:
        return valid_cached

    # 2) 모든 값이 NaN 또는 0인 컬럼은 제거
    non_zero = []
    for c in df_new.columns:
        col = df_new[c]
        # numeric 컬럼: 모두 NaN 혹은 모두 0 이면 삭제
        if pd.api.types.is_numeric_dtype(col):
            if not ((col.isna()) | (col == 0)).all():
                non_zero.append(c)
        else:
            # 비numeric 컬럼은 모두 유지
            non_zero.append(c)
    df_new = df_new[non_zero]

    # 3) valid_cached 컬럼 순서에 맞춰 재색인 (없는 컬럼 자동 NaN)
    df_new = df_new.reindex(columns=valid_cached.columns, fill_value=pd.NA)

    df_new.dropna(axis=1, how='all', inplace=True)  # 모든 값이 NA인 컬럼 제거
    valid_cached.dropna(axis=1, how='all', inplace=True)

    # 4) 이제 컬럼이 일치하므로 concat해도 FutureWarning 없음
    result = pd.concat([valid_cached, df_new], ignore_index=True)
    return result

def cache_quarter_to_db(dfq: pd.DataFrame, engine: Engine):
    """
    staging 테이블에 복사 후 upsert
    """
    dfq.rename(columns=str.lower, inplace=True)
    # actual_date가 비어있는 행만 보정
    # 모든 행에 apply 하지 않고, actual_date가 NaN인 일부 행(mask)만 처리
    # safe_adjust 래퍼를 만들어, 내부에서 adjust_to_trading_day가 실패하면 원본일(YYYYMMDD)을 그대로 반환
    mask = dfq['actual_date'].isna()
    if mask.any():
        def safe_adjust(r):
            q_end = {'1Q':'0331','2Q':'0630','3Q':'0930','FY':'1231'}[r.fiscal_qtr]
            ds = f"{int(r.fiscal_year)}{q_end}"
            try:
                return adjust_to_trading_day(ds)
            except Exception:
                # 보정 불가 시 원래 분기말 날짜 포맷으로 대체
                return ds

        dfq.loc[mask, 'actual_date'] = dfq[mask].apply(safe_adjust, axis=1)    

    # staging 트랜케이트 및 COPY
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE TABLE krx_cache_staging;"))
    # COPY용: shares_outstanding, market_price, fiscal_year를 int로 변환 (소수점 제거)
    dfq['shares_outstanding'] = dfq['shares_outstanding'].astype(int)
    dfq['market_price']      = dfq['market_price'].astype(int)
    dfq['fiscal_year']       = dfq['fiscal_year'].astype(int)
    dfq['shares_outstanding'] = dfq['shares_outstanding'].astype(int)
    dfq['market_price']      = dfq['market_price'].astype(int)
    buf = io.StringIO()
    dfq.to_csv(buf, index=False, header=False, na_rep='')
    buf.seek(0)
    raw = engine.raw_connection(); cur = raw.cursor()
    cur.copy_expert("COPY krx_cache_staging FROM STDIN WITH (FORMAT CSV, NULL '');", buf)
    raw.commit(); cur.close(); raw.close()

    upsert_sql = text(
        """
        INSERT INTO krx_cache AS tgt
        SELECT * FROM krx_cache_staging
        ON CONFLICT (ticker,fiscal_year,fiscal_qtr) DO UPDATE
          SET shares_outstanding=EXCLUDED.shares_outstanding,
              market_price     =EXCLUDED.market_price,
              eps              =EXCLUDED.eps,
              bps              =EXCLUDED.bps,
              per              =EXCLUDED.per,
              pbr              =EXCLUDED.pbr,
              actual_date      =EXCLUDED.actual_date
        """
    )
    with engine.begin() as conn:
        result = conn.execute(upsert_sql)
    # logger.info(f"Upserted {result.rowcount} rows into krx_cache")
