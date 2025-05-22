# generate_fs_summary.py
import os, glob
import pandas as pd

# 1) data/raw/*.csv 파일 목록
raw_files = glob.glob("data/raw/*.csv")

rows = []
for fp in raw_files:
    df = pd.read_csv(fp, dtype=str)
    df['thstrm_amount'] = pd.to_numeric(df['thstrm_amount'], errors='coerce')
    pivot = df.set_index('account_nm')['thstrm_amount'].sum()

    rows.append({
        '회사명':     df['corp_name'].iloc[0],
        '티커':       df['ticker'].iloc[0],
        '연도':       int(df['year'].iloc[0]),
        '매출액':     pivot.get('매출액'),
        '영업이익':   pivot.get('영업이익'),
        '당기순이익': pivot.get('당기순이익'),
        '자산총계':   pivot.get('자산총계'),
        '부채총계':   pivot.get('부채총계'),
        '자본총계':   pivot.get('자본총계'),
    })

# 2) 요약 DataFrame 생성 및 저장
summary_df = pd.DataFrame(rows)
os.makedirs("data", exist_ok=True)
summary_df.to_csv("data/fs_summary.csv", index=False, encoding="utf-8-sig")

print(f"Generated fs_summary.csv ({len(summary_df)} rows)")
