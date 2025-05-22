# app.py
import streamlit as st
from datetime import datetime
import pandas as pd

from src.data_collection.dart_api import fetch_all_corp_codes
from src.utils.db import fetch_dataframe

# 페이지 설정
st.set_page_config(page_title="Stock Analysis Dashboard", layout="wide")

# --- summary_financials 기준 유효 종목 리스트 로드 ---
summary_tickers = fetch_dataframe(
    "SELECT DISTINCT ticker FROM summary_financials", {}
)['ticker'].astype(str).tolist()

# --- 전체 종목 정보 로드 및 필터링 (상장사 + summary 데이터 보유) ---
codes = fetch_all_corp_codes()
df_codes = pd.DataFrame(codes)
df_codes['stock_code'] = df_codes['stock_code'].astype(str).str.zfill(6)
mask = (
    df_codes['stock_code'].str.match(r'^\d{6}$') &
    df_codes['stock_code'].isin(summary_tickers)
)
df_codes = df_codes[mask]
df_codes['label'] = df_codes['stock_code'] + '  ' + df_codes['corp_name']

# --- 사이드바 입력 ---
with st.sidebar:
    st.header("🔎 필터링")

    # 검색 텍스트 입력
    search_text = st.text_input("종목코드 또는 회사명 입력", "")

    # 필터링된 종목 리스트 생성
    filtered_df = df_codes.copy()
    if search_text:
        filtered_df = df_codes[
            df_codes['stock_code'].str.contains(search_text) |
            df_codes['corp_name'].str.contains(search_text)
        ]

    # 종목 선택 콤보박스
    if filtered_df.empty:
        st.warning("조건에 맞는 종목이 없습니다.")
        ticker = corp_name = None
    else:
        selected = st.selectbox(
            "종목 선택",
            options=sorted(filtered_df['label']),
            help="리스트에서 종목을 선택하세요"
        )
        ticker, corp_name = selected.split(maxsplit=1)

    # 연도 선택
    year = st.number_input(
        "연도",
        min_value=2000,
        max_value=datetime.now().year,
        value=datetime.now().year - 1,
        step=1
    )

    # 재무제표 구분 선택
    fs_div_map = {"CFS": "연결재무제표", "OFS": "개별재무제표"}
    fs_div = st.selectbox(
        "재무제표 구분",
        options=list(fs_div_map.keys()),
        format_func=lambda x: f"{x} ({fs_div_map[x]})"
    )

    # 보고서 코드 선택
    reprt_map = {
        "11011": "연간사업보고서",
        "11014": "3분기보고서",
        "11012": "반기보고서",
        "11013": "1분기보고서"
    }
    reprt_code = st.selectbox(
        "보고서",
        options=list(reprt_map.keys()),
        format_func=lambda x: reprt_map[x]
    )

# --- 메인 컨텐츠 ---
st.title("📊 Stock Analysis Dashboard")
st.subheader(
    f"{reprt_map[reprt_code]} ({ticker}) – {corp_name} – {year}년 ({fs_div_map[fs_div]})"
)

# --- 요약 재무 지표 조회 ---
summary_sql = """
SELECT operating_margin         AS "영업이익률(%)",
       roe                     AS "ROE(%)",
       debt_ratio              AS "부채비율(%)",
       controlling_debt_ratio  AS "부채비율(지배주주 기준 %)"
FROM summary_financials
WHERE ticker = :ticker
  AND year   = :year
  AND report_code = :reprt
  AND fs_div      = :fs_div
"""
params = {"ticker": ticker, "year": int(year), "reprt": reprt_code, "fs_div": fs_div}
summary_df = fetch_dataframe(summary_sql, params)
if summary_df.empty:
    st.warning("선택된 조건의 요약 재무 데이터가 없습니다.")
    st.stop()

# 메트릭 표시
def fmt(x):
    return f"{x:,.2f}" if pd.notnull(x) else "N/A"

om = summary_df["영업이익률(%)"].iloc[0]
roe = summary_df["ROE(%)"].iloc[0]
dr = summary_df["부채비율(%)"].iloc[0]
cdr = summary_df["부채비율(지배주주 기준 %)"].iloc[0]
# Debt-to-Equity 비율 계산
if pd.notnull(dr) and dr < 100:
    de_ratio = dr / (100 - dr) * 100
else:
    de_ratio = None
cols = st.columns(5)
cols[0].metric("영업이익률(%)", fmt(om))
cols[1].metric("ROE(%)", fmt(roe), delta="높을수록 효율적 이익")
cols[2].metric("부채비율 (%)", fmt(dr))
cols[3].metric("부채대자본비율 (%)", fmt(de_ratio))
cols[4].metric("지배주주 D/E 비율 (%)", fmt(cdr))

st.markdown("---")

# --- 원본 재무제표 보기 (백만 원 단위) ---
if st.checkbox("원본 재무제표 보기 (백만 원 단위)"):
    raw_sql = """
    SELECT account_nm   AS "계정명",
           (thstrm_amount::numeric/1000000) AS "당기금액(백만 원)",
           (frmtrm_amount::numeric/1000000) AS "전기금액(백만 원)"
    FROM raw_financials
    WHERE ticker = :ticker
      AND year   = :year
      AND report_code = :reprt
      AND fs_div      = :fs_div
    """
    raw_df = fetch_dataframe(raw_sql, params)
    if raw_df.empty:
        st.info("원본 재무제표 데이터가 없습니다.")
    else:
        def to_korean_amt(x):
            if pd.isna(x):
                return ""
            amount = int(x * 1_000_000)
            jo = amount // 1_000_000_000_000
            rem = amount % 1_000_000_000_000
            eok = rem // 100_000_000
            rem2 = rem % 100_000_000
            man = rem2 // 10_000
            thous = man // 1000
            hund = (man % 1000) // 100
            parts = []
            if jo:
                parts.append(f"{jo}조")
            if eok:
                parts.append(f"{eok:,}억")
            if thous:
                parts.append(f"{thous}천")
            if hund:
                parts.append(f"{hund}백")
            if jo or eok or thous or hund:
                parts.append("만 원")
                return " ".join(parts)
            return f"{amount:,}원"
        raw_fmt = raw_df.copy()
        raw_fmt["당기금액(백만 원)"] = raw_fmt["당기금액(백만 원)"].map(lambda x: f"{x:,.0f}" if pd.notnull(x) else "")
        raw_fmt["전기금액(백만 원)"] = raw_fmt["전기금액(백만 원)"].map(lambda x: f"{x:,.0f}" if pd.notnull(x) else "")
        raw_fmt["당기금액(한글)"] = raw_df["당기금액(백만 원)"].apply(to_korean_amt)
        raw_fmt["전기금액(한글)"] = raw_df["전기금액(백만 원)"].apply(to_korean_amt)
        st.dataframe(
            raw_fmt[["계정명", "당기금액(백만 원)", "당기금액(한글)", "전기금액(백만 원)", "전기금액(한글)"]],
            height=500
        )
