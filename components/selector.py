# components/selector.py

import streamlit as st
import pandas as pd
import os

def stock_selector():
    """
    data/financial_summary.csv 에 있는 corp_name, ticker 를
    '삼성전자 (005930)' 형식으로 뽑아서 선택지로 보여줍니다.
    반환: 선택된 '삼성전자 (005930)' 문자열
    """
    path = os.path.join("data", "financial_summary.csv")
    df = pd.read_csv(path, dtype={"ticker": str})
    df["ticker"] = df["ticker"].str.zfill(6)

    # "삼성전자 (005930)" 리스트 생성
    options = df.apply(lambda r: f"{r['corp_name']} ({r['ticker']})", axis=1).tolist()
    selection = st.sidebar.selectbox("종목 선택", options)
    return selection
