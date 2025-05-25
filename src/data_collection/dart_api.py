import sys
import os
import io
import zipfile
import logging
import json
import time
import xml.etree.ElementTree as ET

from datetime import datetime, date, timedelta
from zoneinfo import ZoneInfo
from typing import List, Dict, Tuple

import requests
from sqlalchemy import create_engine, text

from src.utils.db import fetch_dataframe, execute_query
from dotenv import load_dotenv

# ─── 환경 설정 ─────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)
load_dotenv()

DART_API_KEY = os.getenv('DART_API_KEY')
DATABASE_URL = os.getenv('DATABASE_URL')
if not DART_API_KEY or not DATABASE_URL:
    raise RuntimeError("DART_API_KEY 또는 DATABASE_URL이 설정되지 않았습니다.")

engine = create_engine(DATABASE_URL)
kst = ZoneInfo("Asia/Seoul")

# ─── DART OpenAPI 엔드포인트 ────────────────────────────────────────────
CORP_CODE_URL = 'https://opendart.fss.or.kr/api/corpCode.xml'
DART_ENDPOINT  = 'https://opendart.fss.or.kr/api/fnlttSinglAcntAll.json'

REPORT_CODES = ['11011', '11014', '11012', '11013']
REPORT_MAP   = {
    '11011': 'FY',  # 연간
    '11014': '3Q',  # 3분기
    '11012': '2Q',  # 반기
    '11013': '1Q',  # 1분기
}
FS_DIVS = ['CFS', 'OFS']
REVERSE_REPORT_MAP = {v: k for k, v in REPORT_MAP.items()}

# ─── 제한 및 재시도 설정 ─────────────────────────────────────────────────
#MAX_CALLS = int(os.getenv('MAX_CALLS', '19000'))
MAX_CALLS = int(os.getenv('MAX_CALLS', '19200'))
DELAY_BETWEEN_CALLS = float(os.getenv('DART_DELAY', '1.2'))
RETRY_LIMIT = int(os.getenv('DART_RETRY_LIMIT', '3'))
BACKOFF_FACTOR = float(os.getenv('DART_BACKOFF_FACTOR', '1.0'))
SKIP_THRESHOLD = int(os.getenv('DART_SKIP_THRESHOLD', '5'))

# per-corp skip counter
skip_fail_counts: Dict[str, int] = {}

# ─── 유틸리티 ───────────────────────────────────────────────────────────
def get_today_kst() -> date:
    return (datetime.utcnow() + timedelta(hours=9)).date()

# ─── DART API 호출 관리 ─────────────────────────────────────────────────
def fetch(url: str, **kwargs) -> requests.Response:
    today = get_today_kst()
    with engine.begin() as conn:
        conn.execute(text(
            "INSERT INTO dart_state(date, used_calls) VALUES (:d,0) ON CONFLICT(date) DO NOTHING"
        ), {'d': today})
        row = conn.execute(text(
            "UPDATE dart_state SET used_calls = used_calls + 1"
            " WHERE date = :d AND used_calls < :max RETURNING used_calls"
        ), {'d': today, 'max': MAX_CALLS}).fetchone()
    if not row:
        raise RuntimeError(f"DART API 호출 한도({MAX_CALLS}) 초과: {today}")
    try:
        resp = requests.get(url, **kwargs)
        resp.raise_for_status()
        return resp
    except Exception:
        # 호출 실패 시 카운트 복원
        with engine.begin() as conn:
            conn.execute(text(
                "UPDATE dart_state SET used_calls = used_calls - 1 WHERE date = :d"
            ), {'d': today})
        raise

# ─── 법인코드 동기화 ─────────────────────────────────────────────────────
def sync_corp_codes():
    """
    DART 법인코드 ZIP 파일을 항상 다운로드하여
    corp_codes 테이블을 INSERT/UPDATE 처리합니다.
    """
    logger.info("▷ corp_codes 동기화 시작")

    # ZIP 다운로드 & XML 파싱
    resp = fetch(CORP_CODE_URL, params={'crtfc_key': DART_API_KEY}, timeout=30)
    bio = io.BytesIO(resp.content)
    with zipfile.ZipFile(bio) as zf:
        xml_bytes = zf.read(zf.namelist()[0])
    root = ET.fromstring(xml_bytes)

    # UPSERT SQL: stock_code 기준 충돌 시 corp_code, corp_name 업데이트
    sql = """
    INSERT INTO corp_codes(corp_code, stock_code, corp_name)
      VALUES (:corp, :stock, :name)
    ON CONFLICT (stock_code) DO UPDATE
      SET corp_code = EXCLUDED.corp_code,
          corp_name = EXCLUDED.corp_name;
    """

    # 레코드 수집: 빈 stock_code는 제외
    records = []
    for elem in root.findall('list'):
        stock = elem.findtext('stock_code', '').strip()
        if not stock:
            continue
        records.append({
            'corp':  elem.findtext('corp_code', ''),
            'stock': stock.zfill(6),
            'name':  elem.findtext('corp_name', ''),
        })

    # 배치 실행
    with engine.begin() as conn:
        conn.execute(text(sql), records)

    logger.info("▷ corp_codes 동기화 완료")

# ─── 전체 법인코드 조회 ─────────────────────────────────────────────────
def fetch_all_corp_codes() -> List[Dict[str,str]]:
    """
    corp_codes 테이블을 동기화 후 불러옵니다.
    """
    sync_corp_codes()
    df = fetch_dataframe("SELECT corp_code, stock_code, corp_name FROM corp_codes")
    df['stock_code'] = df['stock_code'].astype(str).str.extract(r"(\d+)")[0].str.zfill(6)
    return df.to_dict(orient='records')

# ─── 캐시 upsert 함수 ───────────────────────────────────────────────────
def save_cache(
    corp_code: str,
    stock_code: str,
    year: int,
    fs_qtr: int,
    report_code: str,
    fs_div: str,
    recs: List[Dict]
):
    """dart_cache 테이블에 recs를 upsert"""
    payload = json.dumps(recs, ensure_ascii=False)
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO dart_cache(
              corp_code, stock_code, year, fs_qtr,
              report_code, fs_div, recs
            ) VALUES (
              :c, :s, :y, :q,
              :r, :f, :j
            )
            ON CONFLICT(corp_code, stock_code, year, fs_qtr, report_code, fs_div)
            DO UPDATE SET
              recs         = EXCLUDED.recs,
              last_updated = NOW();
        """), {
            'c': corp_code,
            's': stock_code,
            'y': year,
            'q': fs_qtr,
            'r': report_code,
            'f': fs_div,
            'j': payload,
        })
    # dart자료 api호출하여 캐시 저장 ( 일단 막아놓음 )
    # logger.info(
    #     f"▷ 캐시 저장/업데이트: {corp_name}/ "
    #     f"{year}/ {fs_qtr}/ [{fs_div}/{report_code}]"
    # )    

# ─── 연도별 전체 재무제표 수집 ─────────────────────────────────────────
def fetch_all_statements_for_year(
    corp_code: str,
    stock_code: str,
    year: int,
    rpt: str = None,
    fs_div: str = None
) -> List[Tuple[List[Dict], str, str, str]]:
    """
    corp_code, stock_code, year별로 DART 재무제표(CFS/OFS)를 가져옵니다.
    요청 간 딜레이, 재시도 로직, 클라이언트 호출 수 제한, 반복 실패 시 스킵 처리 포함.
    """
    # 스킵 처리 확인
    if skip_fail_counts.get(corp_code, 0) >= SKIP_THRESHOLD:
        logger.error(f"▷ {corp_code}/{stock_code}: 반복 실패 횟수 초과({SKIP_THRESHOLD})로 스킵 처리")
        return []

    results: List[Tuple[List[Dict], str, str, str]] = []
    report_list = [rpt] if rpt else REPORT_CODES
    fs_list     = [fs_div] if fs_div else FS_DIVS
    skip_for_corp = False

    for rpt_code in report_list:
        for fs in fs_list:
            items = None
            # 재시도 로직
            for attempt in range(1, RETRY_LIMIT + 1):
                # 호출 간 딜레이
                time.sleep(DELAY_BETWEEN_CALLS)
                try:
                    resp = fetch(
                        DART_ENDPOINT,
                        params={
                            'crtfc_key': DART_API_KEY,
                            'corp_code': corp_code,
                            'bsns_year': str(year),
                            'reprt_code': rpt_code,
                            'fs_div': fs,
                        },
                        timeout=15
                    )
                    data = resp.json()

                    # DART 자체 에러 메시지 체크
                    if data.get('status') == 'ERROR':
                        msg = data.get('message', '')
                        if f'한도({MAX_CALLS})' in msg or 'rate limit' in msg.lower():
                            logger.error(f"▷ 호출 한도 초과 감지: {msg}")
                            sys.exit(1)
                    items = data.get('list') or []
                    break

                except Exception as e:
                    # 마지막 재시도 전
                    if attempt < RETRY_LIMIT:
                        backoff = BACKOFF_FACTOR * attempt
                        logger.warning(
                            f"▷ {corp_code}-{stock_code}-{year}-{rpt_code}-{fs} 호출 실패 (재시도 {attempt}/{RETRY_LIMIT}): {e}"
                        )
                        time.sleep(backoff)
                        continue
                    # 마지막 재시도 실패
                    logger.error(
                        f"▷ {corp_code}-{stock_code}-{year}-{rpt_code}-{fs} 호출 실패(재시도 모두 실패): {e}"
                    )
                    skip_fail_counts[corp_code] = skip_fail_counts.get(corp_code, 0) + 1
                    if skip_fail_counts[corp_code] >= SKIP_THRESHOLD:
                        logger.error(
                            f"▷ {corp_code}/{stock_code}: 반복 실패 횟수({skip_fail_counts[corp_code]}) 초과로 스킵 처리"
                        )
                        skip_for_corp = True
                    break

            if skip_for_corp:
                break
            if not items:
                continue

            # 정상 수신 시 파싱
            recs = [
                {
                    'account_id':    it.get('account_id',''),
                    'account_nm':    it.get('account_nm',''),
                    'thstrm_amount': it.get('thstrm_amount',''),
                    'frmtrm_amount': it.get('frmtrm_amount',''),
                    'bfefrm_amount': it.get('bfefrm_amount',''),
                }
                for it in items
            ]
            fiscal_qtr = REPORT_MAP.get(rpt_code, rpt_code)
            results.append((recs, rpt_code, fs, fiscal_qtr))

        if skip_for_corp:
            break

    return results
