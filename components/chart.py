import matplotlib.pyplot as plt
import matplotlib

# 한글폰트 설정 (윈도우: Malgun Gothic, macOS/Linux: NanumGothic 등)
matplotlib.rcParams["font.family"] = "Malgun Gothic"

import pandas as pd

def plot_ratios(df_row: pd.Series):
    """
    df_row: 단일 종목-연도 행. '영업이익률(%)','ROE(%)','부채비율(%)' 컬럼을 갖는다.
    """
    labels = ["영업이익률(%)", "ROE(%)", "부채비율(%)"]
    values = [
        float(df_row["영업이익률(%)"]) if pd.notna(df_row["영업이익률(%)"]) else 0,
        float(df_row["ROE(%)"])        if pd.notna(df_row["ROE(%)"])        else 0,
        float(df_row["부채비율(%)"])    if pd.notna(df_row["부채비율(%)"])    else 0,
    ]

    fig, ax = plt.subplots()
    ax.bar(labels, values)
    ax.set_ylim(0, max(values) * 1.5 if max(values) > 0 else 1)
    ax.set_ylabel("비율 (%)")
    ax.set_title("재무비율 비교")
    return fig
