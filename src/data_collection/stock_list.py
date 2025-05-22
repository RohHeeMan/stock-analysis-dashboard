# src/data_collection/stock_list.py
import requests
import pandas as pd

def fetch_krx_tickers() -> pd.DataFrame:
    url = 'https://kind.krx.co.kr/corpgeneral/corpList.do?method=download&searchType=13'
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    df = pd.read_html(resp.text, header=0)[0]
    df['종목코드'] = df['종목코드'].astype(str).str.zfill(6)
    return df[['회사명','종목코드']].rename(columns={'회사명':'corp_name','종목코드':'stock_code'})
