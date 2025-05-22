import os
from dotenv import load_dotenv
# 이걸 넣으면 .env가 제대로 덮어쓰기 돼서 이전 캐시가 무시될 가능성이 높아져.
load_dotenv(override=True)

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text

# src/utils/db.py
# DATABASE_URL 설정
DATABASE_URL = os.getenv('DATABASE_URL')
if not DATABASE_URL:
    raise RuntimeError("환경변수 DATABASE_URL이 설정되지 않았습니다.")

# SQLAlchemy 엔진 생성
engine = create_engine(DATABASE_URL)


def _convert_params(params: dict) -> dict:
    """
    numpy scalar 타입을 Python 기본 타입으로 변환
    """
    if not params:
        return {}
    converted = {}
    for key, value in params.items():
        if isinstance(value, np.generic):
            converted[key] = value.item()
        else:
            converted[key] = value
    return converted


def fetch_dataframe(query: str, params: dict = None) -> pd.DataFrame:
    """
    SELECT 쿼리를 실행하고 pandas DataFrame으로 반환
    """
    converted = _convert_params(params)
    with engine.connect() as conn:
        result = conn.execute(text(query), converted)
        df = pd.DataFrame(result.fetchall(), columns=result.keys())
    return df


def execute_query(query: str, params: dict = None):
    """
    INSERT/UPDATE/DELETE 쿼리를 트랜잭션 내에서 실행하고 커밋
    """
    converted = _convert_params(params)
    with engine.begin() as conn:
        conn.execute(text(query), converted)
