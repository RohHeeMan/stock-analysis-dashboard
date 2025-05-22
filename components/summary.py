# components/summary.py
import streamlit as st

def show_summary(df):
    st.metric("영업이익률(%)", f"{df['영업이익률(%)'].iloc[0]}")
    st.metric("ROE(%)",       f"{df['ROE(%)'].iloc[0]}")
    st.metric("부채비율(%)",   f"{df['부채비율(%)'].iloc[0]}")
