"""
Query tools generated from financial.csv for financial data queries.
Generated on: 2025-05-20 15:31:44
"""

from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
from langchain.tools import tool


class GetHighestLowestClosePriceByDateStockNumberInput(BaseModel):
    stock_code: str = Field(description="股票代碼")
    acdate: str = Field(description="日期")


@tool("get_highest_lowest_close_prices_by_date_stock_number", args_schema=GetHighestLowestClosePriceByDateStockNumberInput, return_direct=True)
def get_highest_lowest_close_price_by_date_stock_number(
    stock_code: str,
    acdate: str
):
    """
    查詢在xx日期，美股xxxx股票號的最高價、最低價及收市價
    
    Returns:
        str: The query result
    """
    print(stock_code, acdate)
    query = """
  select t1.ACDATE,t1.code,t1.HIGH,t1.LOW,t1.CLOSE
  from SRCIFF.TB_FTS_DAILYPRICEINFO_US t1
  where t1.acdate='{acdate}'
  and t1.CODE='{stock_code}'
    """

    # Format parameters in the query
    query = query.format(
        stock_code=str(stock_code), acdate=str(acdate)
    )

    # Execute query (placeholder for actual execution)
    return f"Executing query: {query}"



class FindStockTechnologyByDateInput(BaseModel):
    stock_code: str = Field(description="股票代碼")
    acdate: str = Field(description="日期")


@tool("find_stock_technology_by_date", args_schema=FindStockTechnologyByDateInput, return_direct=True)
def find_stock_technology_by_date(
    stock_code: str,
    acdate: str
):
    """
    查詢在xx日期，某個xxxx股票號的技術形態
    
    Returns:
        str: The query result
    """
    query = """
  select distinct t1.ACDATE,t1.code,t1.PATTERN_ID,t1.PATTERN_NAME
  from SRCIFF.TB_FTS_DAILYCHARTPATTERN t1
  where t1.acdate='{acdate}'
  and t1.confirmed_date<='{acdate}'
  and t1.expiry_date>'{acdate}'
  and t1.CODE='{stock_code}'
  and t1.pattern_ID in (5,6,7,8,13,14,26,27,30,31,32,34,35,36,41,42,43,44,45,46,50,51,60,61,62,63,70,71)
    """

    # Format parameters in the query
    query = query.format(
        stock_code=str(stock_code), acdate=str(acdate)
    )

    # Execute query (placeholder for actual execution)
    return f"Executing query: {query}"



class FindPredictionRateOnDateAndStockInput(BaseModel):
    company_name: str = Field(description="股票代碼")
    acdate: str = Field(description="日期")


@tool("find_prediction_rate_on_date_and_stock", args_schema=FindPredictionRateOnDateAndStockInput, return_direct=True)
def find_prediction_rate_on_date_and_stock(
    company_name: str,
    acdate: str
):
    """
    查詢在xx日期，某個xx股票(公司名)的預測市盈率
    
    Returns:
        str: The query result
    """
    query = """
  select t1.DATE,t1.code,1/nullifzero(t1.earning_yld_estimate) as estimated_PE ,t3.STOCK_NAME,t3.CHI_NAME
  from SRCIFF.TB_FTS_DAILYCOMPINFO t1 left
  join srciff.TB_DW46SK1 t3 on t3.acdate=t1.acdate
  and t3.stock_code=t1.code
  where t1.acdate='{acdate}'
  and t3.CHI_NAME like '{company_name}'
    """

    # Format parameters in the query
    query = query.format(
        company_name=str(company_name), acdate=str(acdate)
    )

    # Execute query (placeholder for actual execution)
    return f"Executing query: {query}"



class GetTopStocksByDateInput(BaseModel):
    index_code: str = Field(description="股票指數")
    number: int = Field(description="指數中的成分股的數量")
    acdate: str = Field(description="日期")


@tool("get_top_stocks_by_date", args_schema=GetTopStocksByDateInput, return_direct=True)
def get_top_stocks_by_date(
    index_code: str,
    number: int,
    acdate: str
):
    """
    查詢在xx日期，xx指數成分股預測收益率最高的n只股票
    
    Returns:
        str: The query result
    """
    query = """
  select t1.DATE,t1.INDEX_CODE,t1.STOCK_CODE,t1.WEIGHT ,t2.EARNING_YLD_ESTIMATE ,t3.STOCK_NAME,t3.CHI_NAME
  from SRCIFF.TB_FTS_DAILYINDEXCONSTITUENT t1 left
  join SRCIFF.TB_FTS_DAILYCOMPINFO t2 on t2.acdate=t1.acdate
  and t2.date=t1.date
  and t2.code=t1.stock_code left
  join srciff.TB_DW46SK1 t3 on t3.acdate=t1.acdate
  and t3.stock_code=t1.stock_code
  where t1.acdate='{acdate}'
  and t1.index_code='{index_code}' order by t2.EARNING_YLD_ESTIMATE desc
  limit '{number}'
    """

    # Format parameters in the query
    query = query.format(
        index_code=str(index_code), number=str(number), acdate=str(acdate)
    )

    # Execute query (placeholder for actual execution)
    return f"Executing query: {query}"



class FindStocksByDateIndustryInput(BaseModel):
    industry_code: str = Field(description="行業對應代碼")
    index_code: str = Field(description="股票指數代碼")
    acdate: str = Field(description="日期")


@tool("find_stocks_by_date_industry", args_schema=FindStocksByDateIndustryInput, return_direct=True)
def find_stocks_by_date_industry(
    industry_code: str,
    index_code: str,
    acdate: str
):
    """
    查詢在xx日期，xx行業在xx指數有多少隻股票，以及比重
    
    Returns:
        str: The query result
    """
    query = """
  select t1.DATE,t1.INDEX_CODE ,count(t1.STOCK_CODE) as number_of_stock ,sum(t1.WEIGHT) as total_percent
  from SRCIFF.TB_FTS_DAILYINDEXCONSTITUENT t1 left
  join SRCIFF.TB_FTS_DAILYCOMPINFO t2 on t2.acdate=t1.acdate
  and t2.date=t1.date
  and t2.code=t1.stock_code
  where t1.acdate='{acdate}'
  and t1.index_code='{index_code}'
  and (t2.industry='{industry_code}'
  or t2.industry='{industry_code}') group by t1.DATE,t1.INDEX_CODE
    """

    # Format parameters in the query
    query = query.format(
        industry_code=str(industry_code), index_code=str(index_code), acdate=str(acdate)
    )

    # Execute query (placeholder for actual execution)
    return f"Executing query: {query}"



class WorstStockPerformanceOnDateInput(BaseModel):
    index_code: str = Field(description="股票指數代碼")
    acdate: str = Field(description="日期")


@tool("worst_stock_performance_on_date", args_schema=WorstStockPerformanceOnDateInput, return_direct=True)
def worst_stock_performance_on_date(
    index_code: str,
    acdate: str
):
    """
    查詢在xx日期，xxx指數表現最差的股票
    
    Returns:
        str: The query result
    """
    query = """
  select t1.DATE,t1.INDEX_CODE ,t1.STOCK_CODE ,t2.pe_chg_1d ,t3.STOCK_NAME,t3.CHI_NAME
  from SRCIFF.TB_FTS_DAILYINDEXCONSTITUENT t1 left
  join SRCIFF.TB_FTS_DAILYPRICEINFO t2 on t2.acdate=t1.acdate
  and t2.date=t1.date
  and t2.code=t1.stock_code left
  join srciff.TB_DW46SK1 t3 on t3.acdate=t1.acdate
  and t3.stock_code=t1.stock_code
  where t1.acdate= (select max(acdate)
  from SRCIFF.TB_FTS_DAILYINDEXCONSTITUENT
  where acdate <= current_date-7)
  and t1.index_code='{index_code}' order by t2.pe_chg_1d desc
  limit 1
    """

    # Format parameters in the query
    query = query.format(
        index_code=str(index_code), acdate=str(acdate)
    )

    # Execute query (placeholder for actual execution)
    return f"Executing query: {query}"



class FindMinHkStockInput(BaseModel):
    industry_code: str = Field(description="行業對應代碼")
    acdate: str = Field(description="日期")
    number: int = Field(description="篩選股票數量")


@tool("find_min_hk_stock", args_schema=FindMinHkStockInput, return_direct=True)
def find_min_hk_stock(
    industry_code: str,
    acdate: str,
    number: int
):
    """
    查詢xx日期，xx行業PB/ratio最低的n只香港股票
    
    Returns:
        str: The query result
    """
    query = """
  select t1.DATE,t1.CODE,t1.INDUSTRY ,t3.STOCK_NAME,t3.CHI_NAME ,t1.PB_RATIO
  from SRCIFF.TB_FTS_DAILYCOMPINFO t1 left
  join srciff.TB_DW46SK1 t3 on t3.acdate=t1.acdate
  and t3.stock_code=t1.code
  where t1.acdate='{acdate}'
  and t1.industry='{industry_code}'
  and t1.PB_RATIO is not null order by t1.PB_RATIO asc
  limit '{number}'
    """

    # Format parameters in the query
    query = query.format(
        industry_code=str(industry_code), acdate=str(acdate), number=str(number)
    )

    # Execute query (placeholder for actual execution)
    return f"Executing query: {query}"



class FindBlockWeightInput(BaseModel):
    index_code: str = Field(description="股票指數代碼")
    acdate: str = Field(description="日期")
    weight: int = Field(description="權重")


@tool("find_block_weight", args_schema=FindBlockWeightInput, return_direct=True)
def find_block_weight(
    index_code: str,
    acdate: str,
    weight: int
):
    """
    查詢xx日期，在xx指數中比重超過n%的板塊
    
    Returns:
        str: The query result
    """
    query = """
  With ind_con as (
  select t1.ACDATE,t1.INDEX_CODE,t1.STOCK_CODE,t1.WEIGHT ,t2.INDUSTRY
  from SRCIFF.TB_FTS_DAILYINDEXCONSTITUENT t1 left
  join SRCIFF.TB_FTS_DAILYCOMPINFO t2 on t2.acdate=t1.acdate
  and t2.date=t1.date
  and t2.code=t1.stock_code
  where t1.acdate='{acdate}'
  and t1.index_code='{index_code}' )
  select t10.ACDATE,t10.INDUSTRY ,sum(t10.WEIGHT) as industry_weight
  from ind_con t10 group by t10.ACDATE,t10.INDUSTRY
  having sum(t10.WEIGHT)>'{weight}' order by sum(t10.WEIGHT) desc
    """

    # Format parameters in the query
    query = query.format(
        index_code=str(index_code), acdate=str(acdate), weight=str(weight)
    )

    # Execute query (placeholder for actual execution)
    return f"Executing query: {query}"



class GetStockPriceChangeInput(BaseModel):
    start_date: str = Field(description="開始日期")
    end_date: str = Field(description="結束日期")
    index_code: str = Field(description="股票指數代碼")


@tool("get_stock_price_change", args_schema=GetStockPriceChangeInput, return_direct=True)
def get_stock_price_change(
    start_date: str,
    end_date: str,
    index_code: str
):
    """
    查詢xx時間段，xx指數的點數變化
    
    Returns:
        str: The query result
    """
    query = """
  select t1.CODE, sum(change) as agg_change
  from SRCIFF.TB_FTS_DAILYINDEX t1
  where t1.CODE='{index_code}'
  and t1.ACDATE between '{start_date}'
  and '{end_date}' group by t1.CODE
    """

    # Format parameters in the query
    query = query.format(
        start_date=str(start_date), end_date=str(end_date), index_code=str(index_code)
    )

    # Execute query (placeholder for actual execution)
    return f"Executing query: {query}"



class GetPeriodStockReturnInput(BaseModel):
    start_date: str = Field(description="開始日期")
    end_date: str = Field(description="結束日期")
    stock_code: str = Field(description="股票代碼")


@tool("get_period_stock_return", args_schema=GetPeriodStockReturnInput, return_direct=True)
def get_period_stock_return(
    start_date: str,
    end_date: str,
    stock_code: str
):
    """
    查詢xx時間段，xx股票號的回報率
    
    Returns:
        str: The query result
    """
    query = """
  With background_data as (
  select t1.ACDATE,t1.DATE,t1.CODE,t1.SECURITY_TYPE,t1.CLOSE,t1.PER_CHG_1D,t1.TURNOVER ,t1.COUNTER,t1.COMPANY_CODE,t1.CUR,t1.LIST_DATE,coalesce(t1.LIST_DATE,'1970-01-01') as LIST_DATE_DUMMY ,ln(PER_CHG_1D/100+1) as change_1D_log,CASE when t1.CODE=t1.COUNTER then 1 else 2 end as stock_priority
  from SRCIFF.TB_FTS_DAILYPRICEINFO t1
  where acdate<='{end_date}'
  and acdate>='{start_date}'
  and t1.SECURITY_TYPE in ('STK','TRT','TRT_LIP','TRT_REI')-- can it be shown that it does not support this securities type
  and (t1.counter='{stock_code}'
  or t1.code='{stock_code}') order by t1.ACDATE,t1.CODE), last_trading_date_data as(
  select distinct t2.ACDATE,t2.COUNTER,t2.COMPANY_CODE,t2.CUR
  from background_data t2
  where t2.acdate=(select max(acdate)
  from SRCIFF.TB_FTS_DAILYPRICEINFO
  where acdate<='{end_date}'
  and acdate>'{end_date}'::date-7)), stock_selected as(
  select t5.*
  from (select t4.*,rank() over (partition by t4.COMPANY_CODE,t4.COUNTER,t4.ACDATE,t4.DATE order by t4.stock_priority) as ranking
  from ( --- to make sure no reuse RIC is selected by taking out same company_code
  select t3.*
  from background_data t3 inner
  join last_trading_date_data t4 on t4.company_code =t3.company_code
  and t4.counter=t3.counter ) t4 ) t5
  where t5.ranking=1)
  select t7.CODE_ORI,exp (change_cum_log)-1 as percent_chg
  from (select t6.COUNTER as CODE_ORI, sum(t6.change_1D_log) as change_cum_log
  from stock_selected t6 group by t6.COUNTER) t7
    """

    # Format parameters in the query
    query = query.format(
        start_date=str(start_date), end_date=str(end_date), stock_code=str(stock_code)
    )

    # Execute query (placeholder for actual execution)
    return f"Executing query: {query}"



class GetTurnoverByTimeRangeInput(BaseModel):
    start_date: str = Field(description="開始日期")
    end_date: str = Field(description="結束日期")
    stock_code: str = Field(description="股票代碼")


@tool("get_turnover_by_time_range", args_schema=GetTurnoverByTimeRangeInput, return_direct=True)
def get_turnover_by_time_range(
    start_date: str,
    end_date: str,
    stock_code: str
):
    """
    查詢xx時間段，xx股票的turnover
    
    Returns:
        str: The query result
    """
    query = """
  With background_data as (
  select t1.ACDATE,t1.DATE,t1.CODE,t1.SECURITY_TYPE,t1.CLOSE,t1.PER_CHG_1D,t1.TURNOVER ,t1.COUNTER,t1.COMPANY_CODE,t1.CUR,t1.LIST_DATE,coalesce(t1.LIST_DATE,'1970-01-01') as LIST_DATE_DUMMY ,ln(PER_CHG_1D/100+1) as change_1D_log,CASE when t1.CODE=t1.COUNTER then 1 else 2 end as stock_priority
  from SRCIFF.TB_FTS_DAILYPRICEINFO t1
  where acdate<='{end_date}'
  and acdate>='%{start_date}'
  and t1.SECURITY_TYPE in ('STK','TRT','TRT_LIP','TRT_REI')-- can it be shown that it does not support this securities type
  and (t1.counter='{stock_code}'
  or t1.code='{stock_code}') order by t1.ACDATE,t1.CODE), last_trading_date_data as(
  select distinct t2.ACDATE,t2.COUNTER,t2.COMPANY_CODE,t2.CUR
  from background_data t2
  where t2.acdate=(select max(acdate)
  from SRCIFF.TB_FTS_DAILYPRICEINFO
  where acdate<='{end_date}'
  and acdate>'{end_date}'::date-7)), stock_selected as(
  select t5.*
  from (select t4.*,rank() over (partition by t4.COMPANY_CODE,t4.COUNTER,t4.ACDATE,t4.DATE order by t4.stock_priority) as ranking
  from ( --- to make sure no reuse RIC is selected by taking out same company_code
  select t3.*
  from background_data t3 inner
  join last_trading_date_data t4 on t4.company_code =t3.company_code
  and t4.counter=t3.counter ) t4 )t5 --where t5.ranking=1
  select --t6.*
    """

    # Format parameters in the query
    query = query.format(
        start_date=str(start_date), end_date=str(end_date), stock_code=str(stock_code)
    )

    # Execute query (placeholder for actual execution)
    return f"Executing query: {query}"
