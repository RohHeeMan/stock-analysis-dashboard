
import pandas as pd

# src/analysis/ratios.py
# 이 모듈은 raw long-form 재무제표 데이터를 받아 재무지표를 계산합니다.

# **주요 변경 및 설명**

# 1. **capex 로직**: `INVEST_OUT_IDS`와 `INVEST_IN_IDS`를 통해 취득·처분 항목을 분리해 순투자금액을 계산합니다.
# 2. **소수점 반올림**: 
#    - 퍼센트형 지표(`gross_margin` 등)는 `round(2)`로 소수점 둘째 자리까지,
#    - 배수형 지표(`interest_coverage`)는 `round(1)`로 소수점 첫째 자리까지만 저장합니다.
# 3. **가독성 개선**: 주요 섹션(매출, 유동성, 투자활동, 비율, YoY, 반올림 등)마다 주석과 빈 줄을 넣어 가독성을 높였습니다.

############### CFS 만 사용한 이유 ###############

# 1. 현금흐름 계정의 존재 유무
# IFRS 기준으로 영업·투자·재무활동 현금흐름은 오직 Cash Flow Statement(현금흐름표)에만 나옵니다.

# 그리고 Cash Flow Statement 데이터는 Consolidated(연결) 재무제표(CFS)에만 제공되고,
# OFS(별도재무제표) 에는 보통 포함되지 않거나 항목 구성이 다릅니다.

# 그래서 cfo, capex, free_cash_flow 같은 주요 현금흐름 항목을 뽑으려면
# 반드시 CFS 쪽 데이터만 걸러내야 합니다.

# 2. 분석·비교의 일관성
# 연결재무제표(CFS)는 모기업 + 자회사 전체 실적을 합산한 것이어서,
# “회사 전체” 성과를 대표하는 지표로 쓰기에 적합합니다.

# 반면 별도재무제표(OFS)는 모기업 단일 법인 실적만 담고 있어서,
# 그룹 차원의 투자·영업 실적을 비교·분석할 때는 일관된 기준이 되지 않습니다.

# 머신러닝 모델에 넣을 피처를 만들 때도,
# “같은 기준(연결)”의 숫자만 골라 쓰는 게 학습 안정성과 해석 관점에서 모두 유리합니다.

def summary_financials(raw: pd.DataFrame) -> pd.DataFrame:
    """
    raw: DataFrame with columns:
      - ticker, fiscal_year, fiscal_qtr
      - account_id, account_nm, amount (numeric)

    반환 컬럼:
      ticker, fiscal_year, fiscal_qtr,
      revenue, gross_profit, operating_income, net_income,
      gross_margin, operating_margin, net_margin,
      roe, debt_to_equity, fcf_margin,
      cfo, capex, free_cash_flow,
      current_ratio, quick_ratio, interest_coverage, roa,
      rev_yoy, op_inc_yoy, net_inc_yoy,
      shares_outstanding, market_price, eps, bps, per, pbr
    """
    # 1) 반환 컬럼 목록 정의
    cols = [
        'ticker', 'fiscal_year', 'fiscal_qtr',
        'revenue', 'gross_profit', 'operating_income', 'net_income',
        'gross_margin', 'operating_margin', 'net_margin',
        'roe', 'debt_to_equity', 'fcf_margin',
        'cfo', 'capex', 'free_cash_flow',
        'current_ratio', 'quick_ratio', 'interest_coverage', 'roa',
        'rev_yoy', 'op_inc_yoy', 'net_inc_yoy',
        'shares_outstanding', 'market_price', 'eps', 'bps', 'per', 'pbr'
    ]

    # 2) 빈 데이터 확인
    if raw.empty:
        return pd.DataFrame(columns=cols)

    # 3) 헬퍼 함수 정의
    def sum_accts(grp, acct_ids):
        return grp.loc[grp['account_id'].isin(acct_ids), 'amount'].sum()

    def sum_by_name(grp, keywords):
        pattern = '|'.join(keywords)
        mask = grp['account_nm'].str.contains(pattern, na=False)
        return grp.loc[mask, 'amount'].sum()

    # 4) 투자활동 계정 ID 목록 (투자 지출/수입 분해)
    INVEST_OUT_IDS = [
        'ifrs-full_PaymentsToAcquirePropertyPlantAndEquipment',
        'ifrs-full_PaymentsToAcquireIntangibleAssets',
        'ifrs-full_PaymentsToAcquireSubsidiariesAndAssociates',
        # DART 표준계정 예시 추가
        'dart_PurchaseOfInvestmentsInSubsidiariesJointVenturesAndAssociates',
        'dart_PurchaseOfInterestsInInvestmentsAccountedForUsingEquityMethod',
    ]
    INVEST_IN_IDS = [
        'ifrs-full_ProceedsFromDisposalOfPropertyPlantAndEquipment',
        'ifrs-full_ProceedsFromDisposalOfIntangibleAssets',
        'ifrs-full_ProceedsFromDisposalOfSubsidiariesAndAssociates',
        # DART 표준계정 예시 추가
        'dart_ProceedsFromSalesOfInvestmentsInSubsidiariesJointVenturesAndAssociates',
        'dart_ProceedsFromSalesOfInvestmentsAccountedForUsingEquityMethod',
    ]

    # 5) 그룹별 집계
    records = []
    grouped = raw.groupby(['ticker', 'fiscal_year', 'fiscal_qtr'], dropna=False)

    for (tkr, yr, fq), grp in grouped:
        # -- 매출 및 이익 --
        # revenue        = sum_accts(grp, ['ifrs-full_Revenue','ifrs-full_SalesRevenue'])
        # -- 매출 및 이익 --
        # Revenue: 다양한 매출 과목을 모두 포함
        # -- 1. 매출 항목 (넓게 잡기) --
        revenue        = sum_accts(grp, [
            'ifrs-full_Revenue',             # 기본 매출
            'ifrs-full_SalesRevenueGoods',   # 재화 판매 매출
            'ifrs-full_NetSales'             # 순매출
            'dart_SalesRevenue'              # DART에서 가끔 이걸로 제공(신규추가)
        ])        
        
        # -- 2. 매출원가 --
        cost_of_sales  = sum_accts(grp, ['ifrs-full_CostOfSales','ifrs-full_CostOfRevenue'])
        # -- 3. 매출총이익 --
        gross_profit   = sum_accts(grp, ['ifrs-full_GrossProfit','ifrs-full_GrossProfitLoss'])

        if not revenue and gross_profit == 0:
            gross_profit = revenue - cost_of_sales

        # -- 4. 영업이익 (Operating Income) --
        operating_income = sum_accts(grp, [
            'ifrs-full_OperatingIncome',
            'ifrs-full_ProfitLossFromOperatingActivities',
            'dart_OperatingIncomeLoss'
        ])

        # 값이 없거나 0일 경우 gross_profit - op_expenses 방식으로 대체
        if operating_income == 0:
            op_expenses = sum_accts(grp, [
                'ifrs-full_OperatingExpenses',
                'ifrs-full_SellingGeneralAdministrativeExpenses',
                'ifrs-full_AdministrativeExpenses',
                'ifrs-full_SellingExpenses'
            ])

            if gross_profit != 0 and op_expenses != 0:
                operating_income = gross_profit - op_expenses

        # -- 5. 당기순이익 (Net Income) --
        net_income = sum_accts(grp, [
            'ifrs-full_NetIncome',
            'ifrs-full_ProfitLoss'
        ])

        # 없으면 대체 계정으로 계산
        if net_income == 0:
            net_income = sum_accts(grp, [
                'ifrs-full_ProfitLossFromContinuingOperations',
                'dart_ProfitLossFromContinuingOperations'
            ])

        # -- 유동성 --
        current_assets = sum_accts(grp, ['ifrs-full_CurrentAssets'])
        current_liabs  = sum_accts(grp, ['ifrs-full_CurrentLiabilities'])
        inventory      = sum_accts(grp, ['ifrs-full_Inventory'])

        # -- 자본·부채·자산 --
        total_liabs = sum_accts(grp, ['ifrs-full_TotalLiabilities'])
        if total_liabs == 0:
            total_liabs = sum_by_name(grp, ['Liabilities','부채'])

        equity = sum_accts(grp, ['ifrs-full_Equity'])
        if equity == 0:
            equity = sum_by_name(grp, ['Equity','자본총계'])

        total_assets = sum_accts(grp, ['ifrs-full_TotalAssets'])
        if total_assets == 0:
            total_assets = sum_by_name(grp, ['Assets','자산총계'])

        # -- 이자 비용 --
        interest_exp = sum_accts(grp, [
            'ifrs-full_InterestExpense',
            'ifrs-full_FinanceCosts',
            'ifrs-full_FinanceExpense'
        ])

        # -- 영업활동 현금흐름 --
        cfo = sum_accts(grp, [
            'ifrs-full_CashFlowsFromOperatingActivities',
            'ifrs-full_CashFlowsFromUsedInOperatingActivities'
        ])

        # -- 투자활동 현금흐름 (capex) --
        capex_out = sum_accts(grp, INVEST_OUT_IDS)
        capex_in  = sum_accts(grp, INVEST_IN_IDS)
        capex     = capex_out - capex_in

        # -- 잉여현금흐름 --
        free_cash_flow = cfo - capex

        # -- 비율 계산 --
        gross_margin      = (gross_profit    / revenue    * 100) if revenue    else None
        operating_margin  = (operating_income/ revenue    * 100) if revenue    else None
        net_margin        = (net_income      / revenue    * 100) if revenue    else None
        fcf_margin        = (free_cash_flow  / revenue    * 100) if revenue    else None
        current_ratio     = (current_assets  / current_liabs * 100) if current_liabs else None
        quick_ratio       = ((current_assets - inventory) / current_liabs * 100) if current_liabs else None
        debt_to_equity    = (total_liabs     / equity     * 100) if equity     else None
        interest_coverage = (operating_income/ abs(interest_exp)) if interest_exp else None
        roe               = (net_income      / equity     * 100) if equity     else None
        roa               = (net_income      / total_assets* 100) if total_assets else None

        # -- 레코드 추가 --
        records.append({
            'ticker': tkr,
            'fiscal_year': yr,
            'fiscal_qtr': fq,
            'revenue': revenue,
            'gross_profit': gross_profit,
            'operating_income': operating_income,
            'net_income': net_income,
            'gross_margin': gross_margin,
            'operating_margin': operating_margin,
            'net_margin': net_margin,
            'roe': roe,
            'debt_to_equity': debt_to_equity,
            'fcf_margin': fcf_margin,
            'cfo': cfo,
            'capex': capex,
            'free_cash_flow': free_cash_flow,
            'current_ratio': current_ratio,
            'quick_ratio': quick_ratio,
            'interest_coverage': interest_coverage,
            'roa': roa,
            'rev_yoy': None,
            'op_inc_yoy': None,
            'net_inc_yoy': None,
            'shares_outstanding': None,
            'market_price': None,
            'eps': None,
            'bps': None,
            'per': None,
            'pbr': None
        })

    # 6) DataFrame 생성 및 YoY 계산
    df = pd.DataFrame(records)
    if df.empty:
        return df[cols]

    df = df.sort_values(['ticker','fiscal_qtr','fiscal_year'])
    df['rev_yoy']     = df.groupby(['ticker','fiscal_qtr'])['revenue'].pct_change()     * 100
    df['op_inc_yoy']  = df.groupby(['ticker','fiscal_qtr'])['operating_income'].pct_change()* 100
    df['net_inc_yoy'] = df.groupby(['ticker','fiscal_qtr'])['net_income'].pct_change()     * 100
    df[['rev_yoy','op_inc_yoy','net_inc_yoy']] = df[['rev_yoy','op_inc_yoy','net_inc_yoy']].fillna(0)

    # 7) 반올림 처리
    #  - 퍼센트형 지표: 소수점 둘째 자리
    pct_cols = [
        'gross_margin','operating_margin','net_margin',
        'current_ratio','quick_ratio','debt_to_equity',
        'roe','roa','fcf_margin',
        'rev_yoy','op_inc_yoy','net_inc_yoy'
    ]
    df[pct_cols] = df[pct_cols].round(2)

    #  - 배수형 지표: 소수점 첫째 자리
    df['interest_coverage'] = df['interest_coverage'].round(1)

    # 8) 최종 반환
    return df[cols]


