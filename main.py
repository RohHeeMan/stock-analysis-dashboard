import os
import json   # load_cached에서 JSON 파싱용aa
import warnings
import logging
import pandas as pd
import sys
from sqlalchemy import create_engine, text
from datetime import datetime
from zoneinfo import ZoneInfo
from dotenv import load_dotenv

# 분석 모듈
from src.analysis.ratios import summary_financials
# DART API 모듈
from src.data_collection.dart_api import (
    save_cache,
    fetch_all_corp_codes,
    fetch_all_statements_for_year,
    REPORT_CODES,
    FS_DIVS,
)
# KRX API 및 캐시 모듈
from src.data_collection.krx_api import (
    fetch_quarter_data,
    cache_quarter_to_db,
)

# --- 환경 및 로깅 설정 ---
load_dotenv(override=True)
DATABASE_URL = os.getenv('DATABASE_URL') or sys.exit('DATABASE_URL 미설정')
logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# --- DB 접속, 타임존 ---
engine = create_engine(DATABASE_URL)
kst = ZoneInfo("Asia/Seoul")

# --- 날짜 및 분기 동적 설정 ---
today = datetime.now(kst)

# 과거 연도 포함: (현재연도-3 : 3년치 ) ~ 현재연도
YEARS = list(range(today.year - 3, today.year + 1))
# 3년치 계산시 raw_financials의 전기/전전기금액을 가져오기 위한 최저치 년도.
LIMIT_API_YEAR = min(YEARS)-2  # 전기/전전기
ALL_QTRS = ['1Q','2Q','3Q','FY']

def available_quarters_for(year: int) -> list[str]:
    if year < today.year:
        return ALL_QTRS.copy()
    qtrs = []
    m, d = today.month, today.day
    if (m, d) >= (4, 1):
        qtrs.append('1Q')
    if (m, d) >= (7, 1):
        qtrs.append('2Q')
    if (m, d) >= (10, 1):
        qtrs.append('3Q')
    # FY(사업보고서)는 이듬해 3월 말 이후 발표되므로 올해에는 아직 없음
    return qtrs

# 올해 연도 포함 : API호출을 줄이기 위해 ( 1Q, 2Q, 3Q, 4Q등 올해 공시 되지 않은 쿼터는 제외하고 호출하기 위해서 )
def available_report_codes_for(year: int) -> list[str]:
    """
    주어진 연도에 대해 현재 시점에서 발표되었을 가능성이 있는 분기 report_code 리스트를 반환.
    - 1월~4월: []  (아직 분기보고서 미발표)
    - 5월~7월: ['11013']  # 1Q
    - 8월~10월: ['11013','11012']  # 1Q,2Q
    - 11월~12월: ['11013','11012','11014']  # 1Q,2Q,3Q
    - 내년 연초(1~4월): 올해 FY('11011')도 가능
    """
    today = datetime.now(kst)
    if year < today.year:
        return REPORT_CODES[:]  # 과거 연도는 모두
    # 올해
    codes = []
    if today.month >= 5:
        codes.append('11013')  # 1Q
    if today.month >= 8:
        codes.append('11012')  # 2Q
    if today.month >= 11:
        codes.append('11014')  # 3Q
    # FY는 내년 1~4월에 발표되므로, 그 시점에만 추가
    if year < today.year or (year == today.year and today.month >= 1 and today.month <= 4):
        codes.append('11011')
    return codes

# --- DART·KRX 공통 설정 ---
# 보고서코드→분기 매핑
# --- DART·KRX 공통 설정 ---
# 11011 은 “사업보고서” 또는 “연간 실적”(FY)을 의미합니다.
# 11013: 1분기보고서 (1Q)
# 11012: 반기보고서 (2Q)
# 11014: 3분기보고서 (3Q)
REPORT_MAP = {'11011':'FY','11014':'3Q','11012':'2Q','11013':'1Q'}

# 올해 분기 가능한 재무제표를 가져오고 필요없는 분기 API는 호출하지 않기 위해 사용
REVERSE_REPORT_MAP = {v: k for k, v in REPORT_MAP.items()}

# --- DART 캐시 조회 함수 ---
def load_cached(corp_code: str, stock_code: str, year: int, fs_div: str, fs_qtr: str, report_code: str):
    with engine.connect() as conn:
        row = conn.execute(text(
            """
            SELECT recs
              FROM dart_cache
             WHERE corp_code   = :c
               AND stock_code  = :s
               AND year        = :y
               AND fs_div      = :f
               AND fs_qtr      = :q
               AND report_code = :r
            """
        ), {'c': corp_code,'s': stock_code,'y': year,'f': fs_div,'q': fs_qtr,'r': report_code}).fetchone()
    if not row or not row[0]:
        return None
    if isinstance(row[0], (list, dict)):
        return row[0]
    try:
        return json.loads(row[0])
    except Exception:
        return None

# --- summary_financials upsert 함수 ---
def upsert_summary(df_summary: pd.DataFrame, engine, logger):
    """
    summary_financials 테이블에 upsert 수행.
    PK: (ticker, fiscal_year, fiscal_qtr, fs_div)
    
    summary_financials 테이블에 upsert 수행.
    PK: (ticker, fiscal_year, fiscal_qtr, fs_div)
    """    
    df = df_summary.copy()
    df['fs_div'] = 'CFS'
    # ③ 업서트할 컬럼 리스트 (테이블 스키마와 정확히 일치)
    cols = [
        'ticker','fiscal_year','fiscal_qtr','fs_div',
        'revenue','gross_profit','operating_income','net_income',
        'cfo','capex','free_cash_flow',
        'gross_margin','operating_margin','net_margin',
        'current_ratio','quick_ratio','debt_to_equity',
        'interest_coverage','roe','roa','fcf_margin',
        'rev_yoy','op_inc_yoy','net_inc_yoy',
        'shares_outstanding','market_price','eps','bps','per','pbr',
    ]
    insert_sql = text(f"""
    INSERT INTO summary_financials ({','.join(cols)})
    VALUES ({','.join(':'+c for c in cols)})
    ON CONFLICT (ticker, fiscal_year, fiscal_qtr, fs_div) DO UPDATE SET
      {', '.join(f"{c} = EXCLUDED.{c}" for c in cols if c not in ['ticker','fiscal_year','fiscal_qtr','fs_div'])}
    """
    )
    with engine.begin() as conn:
        records = df[cols].to_dict(orient='records')
        for rec in records:
            conn.execute(insert_sql, rec)
    logger.info(f"✓ SUMMARY_FINANCIALS upsert 완료: {len(records)}건")

def has_data_in_db(tkr, year, fs_qtr, rpt, fs_div):
    with engine.connect() as conn:
        count = conn.execute(text("""
            SELECT COUNT(*) FROM raw_financials
            WHERE ticker = :tk AND fs_year = :yr AND fs_qtr = :fq
              AND report_code = :rp AND fs_div = :fd
        """), {'tk': tkr, 'yr': year, 'fq': fs_qtr, 'rp': rpt, 'fd': fs_div}).scalar()
    return count > 0

# --- main 파이프라인 ---
def main():
    cache_hits = 0
    api_calls = 0
    start = datetime.now(kst)
    logger.info(f"[시작] 전체 파이프라인 - {start.isoformat()}")

    # 1) corp_codes 로드
    try:
        df_codes = pd.DataFrame(fetch_all_corp_codes())
        if df_codes.empty:
            logger.error("corp_codes 테이블이 비어 있습니다.")
            sys.exit(1)
        raw_targets = os.getenv('TARGET_TICKERS', '')

        targets = [t.strip().zfill(6) for t in raw_targets.split(',') if t.strip()]
        if targets:
            # df_codes 를 필터링
            df_codes = df_codes[df_codes['stock_code'].isin(targets)]
            # 각 티커별로 corp_name을 한 줄씩 로깅
            for _, row in df_codes.iterrows():
                logger.info(f"▷ TARGET_TICKER: {row['stock_code']} → {row['corp_name']}")
                
        # 2) RAW 데이터 수집 (지난 3년 + 올해)
        raw_sql = text(
            """
            INSERT INTO raw_financials(
            corp_name, ticker, fs_year, fs_qtr, report_code, fs_div,
            account_id, account_nm,
            thstrm_amount, frmtrm_amount, bfefrm_amount,
            created_at
            ) VALUES (
            :cn, :tk, :yr, :fq, :rp, :fd,
            :aid, :anm,
            :ta, :fa, :ba,
            NOW()
            ) ON CONFLICT(ticker,fs_year,fs_qtr,report_code,fs_div,account_id) DO NOTHING;
            """
        )

        for _, row in df_codes.iterrows():
            corp_code = row['corp_code']
            tkr       = row['stock_code']
            corp_name = row['corp_name']

            for yr in YEARS:
                prev_year = yr - 1

                rpt_list = available_report_codes_for(yr)
                if not rpt_list:
                    logger.info(f"▷ {yr}년에는 아직 수집할 분기보고서가 없습니다.")
                    continue

                logger.info(f"▶ DART 수집: {tkr}|{corp_name}|{yr} → report_codes: {rpt_list}")
                results = []

                for rpt in rpt_list:
                    fs_qtr = REPORT_MAP[rpt]
                    for fs_div in FS_DIVS:
                        cached = load_cached(corp_code, tkr, yr, fs_div, fs_qtr, rpt)
                        if cached is not None and len(cached) > 0:
                            cache_hits += 1
                            with engine.begin() as conn:
                                for r in cached:
                                    conn.execute(raw_sql, {
                                        'cn': corp_name, 'tk': tkr,
                                        'yr': yr, 'fq': fs_qtr, 'rp': rpt, 'fd': fs_div,
                                        'aid': r.get('account_id'),
                                        'anm': r.get('account_nm'),
                                        'ta': int(float(r.get('thstrm_amount') or 0)),                                        
                                        'fa': int(float(r.get('frmtrm_amount') or 0)),
                                        'ba': int(float(r.get('bfefrm_amount') or 0))
                                    })
                            continue

                        # 캐시 없으면 DB raw_financials 존재여부 체크(전기금액,전전기금액 호출하기 전 존재유무 판단)
                        # 3년치 데이터를 기준으로 가져오는데 예를들면 2022년도까지만 있고 2021은 당연히 데이터가 없다.
                        # 그런데 전년도 전전년도 금액이 2022년 데이터의 필드로 있으니까 과거 데이터를 찾으면 안되고
                        # 2022년도의 raw_financials 테이블의 frmtrm_amount,bfefrm_amount를 통해 가져와야 한다.
                        # 그래야 이미 집계된 자료의 호출을 줄일수 있다.
                        with engine.connect() as conn:
                            count = conn.execute(text("""
                                SELECT COUNT(*) FROM raw_financials
                                WHERE ticker = :tk AND fs_year = :yr AND fs_qtr = :fq
                                AND report_code = :rp AND fs_div = :fd
                            """), {'tk': tkr, 'yr': yr, 'fq': fs_qtr, 'rp': rpt, 'fd': fs_div}).scalar()

                        if count > 0:
                            continue  # DB에 데이터가 있으면 API 호출 안 함

                        api_calls += 1
                        try:
                            part = fetch_all_statements_for_year(corp_code, tkr, yr, rpt, fs_div)
                            results.extend(part)
                        except Exception as e:
                            logger.warning(f"{tkr}-{yr}-{fs_div}/{rpt} API 호출 실패: {e}")

                # --- 전전기 데이터 수집 (prev_year) ---
                prev = []
                prev_qtrs = available_quarters_for(prev_year)
                prev_rpt_list = [REVERSE_REPORT_MAP[q] for q in prev_qtrs]

                if prev_year >= LIMIT_API_YEAR:
                    for rpt in prev_rpt_list:
                        fs_qtr = REPORT_MAP[rpt]
                        for fs_div in FS_DIVS:
                            cached_prev = load_cached(corp_code, tkr, prev_year, fs_div, fs_qtr, rpt)

                            if cached_prev is not None and len(cached_prev) > 0:
                                prev.append((cached_prev, rpt, fs_div, fs_qtr))
                                continue

                            # DB 체크: 해당 행(row)의 frmtrm_amount 또는 bfefrm_amount 중 적어도 하나에 데이터가 있어야만 조건을 만족
                            with engine.connect() as conn:
                                count_next_year = conn.execute(text("""
                                    SELECT COUNT(*) FROM raw_financials
                                    WHERE ticker = :tk AND fs_year = :next_yr AND fs_qtr = :fq
                                    AND report_code = :rp AND fs_div = :fd
                                    AND (frmtrm_amount IS NOT NULL OR bfefrm_amount IS NOT NULL)
                                """), {'tk': tkr, 'next_yr': prev_year + 1, 'fq': fs_qtr, 'rp': rpt, 'fd': fs_div}).scalar()

                            if count_next_year > 0:
                                # 2022년 데이터가 있으니 2021년 데이터는 API 호출하지 않아도 됨
                                # 값이 있으면 Pass한다.
                                continue

                            api_calls += 1
                            try:
                                part = fetch_all_statements_for_year(corp_code, tkr, prev_year, rpt, fs_div)
                                prev.extend(part)
                            except Exception as e:
                                logger.warning(f"{tkr}-{prev_year}-{fs_div}/{rpt} PREV API 호출 실패: {e}")
                else:
                    logger.debug(f"[SKIP PREV DATA] {tkr} - {prev_year} 이전 연도는 API 호출하지 않음.")


                # bfefrm_amount 보정을 위한 prev_map 생성
                prev_map = {
                    (pr['account_id'], fs_div, rpt): pr.get('bfefrm_amount') or pr.get('thstrm_amount')
                    for recs, rpt, fs_div, fs_qtr in prev
                    for pr in (recs if isinstance(recs, list) else [])
                }

                if not results:
                    continue

                # bfefrm_amount 보정 및 DB 저장
                for recs, rpt, fs_div, fs_qtr in results:
                    fs_qtr = REPORT_MAP[rpt]
                    for r in recs:
                        if not r.get('bfefrm_amount'):
                            key = (r['account_id'], fs_div, rpt)
                            if key in prev_map and prev_map[key] is not None:
                                r['bfefrm_amount'] = prev_map[key]

                    if load_cached(corp_code, tkr, yr, fs_div, fs_qtr, rpt) is None:
                        save_cache(corp_code, tkr, yr, fs_qtr, rpt, fs_div, recs)
                        with engine.begin() as conn:
                            for r in recs:
                                conn.execute(raw_sql, {
                                    'cn': corp_name,
                                    'tk': tkr,
                                    'yr': yr,
                                    'fq': fs_qtr,
                                    'rp': rpt,
                                    'fd': fs_div,
                                    'aid': r.get('account_id'),
                                    'anm': r.get('account_nm'),
                                    'ta': int(float(r.get('thstrm_amount') or 0)),                                        
                                    'fa': int(float(r.get('frmtrm_amount') or 0)),
                                    'ba': int(float(r.get('bfefrm_amount') or 0))
                                })
    except RuntimeError as e:
        if "DART API 호출 한도" in str(e):
            print(f"[중단] {e}")
            return  # 여기서 중단
        else:
            raise  # 그 외는 다시 에러 발생

    # 3) SUMMARY 생성
    logger.info("▶ STEP1: SUMMARY 생성 중...")
    df_raw = pd.read_sql_query(
        text("""
            SELECT ticker, fs_year AS fiscal_year,
                   report_code, fs_div,
                   account_id, account_nm,
                   thstrm_amount::numeric AS amount,
                   created_at AS report_date
              FROM raw_financials
             WHERE fs_year BETWEEN :start AND :end
               AND fs_div IN ('CFS','OFS')
        """),
        engine,
        params={'start': YEARS[0], 'end': YEARS[-1]}
    )
    df_raw['div_pri'] = df_raw['fs_div'].map({'CFS': 0, 'OFS': 1})
    df_raw = (
        df_raw
        .sort_values(['ticker','fiscal_year','report_code','account_id','div_pri'])
        .drop_duplicates(subset=['ticker','fiscal_year','report_code','account_id'], keep='first')
        .drop(columns='div_pri')
    )
    df_raw['fiscal_qtr'] = df_raw['report_code'].map(REPORT_MAP)
    df_summary = summary_financials(df_raw)
    logger.info(f"✓ STEP1: SUMMARY 생성 완료 ({len(df_summary)}건)")

    # 4) summary_financials 업서트
    logger.info(f"▶ STEP2: summary_financials 테이블 업서트 중: {corp_name}")
    upsert_summary(df_summary, engine, logger)

    # 4) KRX 캐시 작업: 연도별 1Q·2Q·3Q·FY 네 분기를 한 번에
    for year in YEARS:
        qs = available_quarters_for(year)
        if not qs:
            continue
        logger.info(f"▷ KRX CSV자료 다운로드: {year} | 분기 {'/'.join(qs)} | {corp_name}")
        dfs = []
        for qtr in qs:
            # Fetching 로그 표시 (일단 막음)
            #logger.info(f"  · Fetching {year} {qtr} …")
            part = fetch_quarter_data(year, qtr, df_codes, engine)
            if not part.empty:
                dfs.append(part)
        if not dfs:
            logger.info(f"⚠️ {year}년 수집 대상 없음: {corp_name}")
            continue
        dfq = pd.concat(dfs, ignore_index=True)
        cache_quarter_to_db(dfq, engine)
        logger.info(f"✓ KRX 캐시 저장 완료: {corp_code} {corp_name} | {year}년 ({len(dfq)}건)")

    # 5) 캐시에 저장된 전체 분기 데이터를 다시 불러와서 df_val로 정의
    df_val = pd.read_sql_query(
        text("""
            SELECT
            ticker,
            fiscal_year,
            fiscal_qtr,
            shares_outstanding,
            market_price,
            eps,
            bps,
            per,
            pbr
            FROM krx_cache
        """),
        engine
    )

    # 6) 이어서 df_summary 와 df_val 병합
    df_merged = df_summary.merge(
        df_val,
        on=['ticker','fiscal_year','fiscal_qtr'],
        how='left'
    )

    # 기존 placeholder 컬럼 삭제 (merge 시 _x/_y 생기는 문제 방지)
    df_summary.drop(columns=[
        'shares_outstanding','market_price','eps','bps','per','pbr'
   ], errors='ignore', inplace=True)

    df_summary = df_summary.merge(df_val, on=['ticker','fiscal_year','fiscal_qtr'], how='left')
    logger.info("✓ STEP2: KRX 데이터 병합 완료")

    # 재무제표 보는법
    # FY - (1Q + 2Q + 3Q) = 4Q
    # 네이버의 2024기준
    # 4분기 매출 = 연간 매출 - 3분기 누적 매출
    # = 107,380억 원 - (25,261 + 26,105 + 27,156)억 원
    # = 107,380억 원 - 78,522억 원
    # = 28,858억 원

    # summary_financials에 KRX 지표 업데이트
    logger.info("▶ STEP3: summary_financials에 KRX 지표 업데이트 중...")
    with engine.begin() as conn:
        conn.execute(text("""
            UPDATE summary_financials AS sf
            SET
            shares_outstanding = k.shares_outstanding,
            market_price       = k.market_price,
            eps                = k.eps,
            bps                = k.bps,
            per                = k.per,
            pbr                = k.pbr
            FROM krx_cache AS k
            WHERE sf.ticker      = k.ticker
            AND sf.fiscal_year = k.fiscal_year
            AND sf.fiscal_qtr  = k.fiscal_qtr;
        """))
    logger.info("✓ summary_financials에 KRX 지표 업데이트 완료")
    end = datetime.now(kst)
    logger.info(f"[완료] {end.isoformat()} (소요: {end-start})")

if __name__ == '__main__':
    try:
        main()
    except Exception:
        logger.exception("스크립트 에러")
        sys.exit(1)
