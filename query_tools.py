"""
Query tools generated from financial.csv for financial data queries.
Generated on: 2025-04-16 09:32:28
"""

from typing import Optional, Dict, Any, List, Union
from langchain.tools import BaseTool

# Parameter descriptions collected from CSV and defaults
PARAMETER_DESCRIPTIONS = {
    "acdate": "日期",
    "company_name": "股票代碼",
    "end_date": "結束日期",
    "index_code": "股票指數",
    "industry_code": "行業對應代碼",
    "number": "指數中的成分股的數量",
    "start_date": "開始日期",
    "stock_code": "股票代碼",
    "weight": "權重",
}



def query_get_highest_lowest_price(stock_code: str = '5', acdate: str = '2023-03-01') -> str:
    """
    查詢在xx日期，美股xxxx股票號的最高價、最低價及收市價

    Parameters:
        stock_code (str): 股票代碼
        acdate (str): 日期

    Returns:
        str: The query result
    """
    query = """
        select  			t1.ACDATE,t1.code,t1.HIGH,t1.LOW,t1.CLOSE  			from SRCIFF.TB_FTS_DAILYPRICEINFO_US t1 			where t1.acdate='{acdate}' 			and t1.CODE='{stock_code}'
    """

    # Correct replacement using the actual parameter names in curly braces
    query = query.format(
        stock_code=str(stock_code) if stock_code is not None else "", acdate=str(acdate) if acdate is not None else ""
    )

    # Execute query
    return f"Executing query: {query}"



def query_get_stock_technology_by_date(stock_code: str = '5', acdate: str = '2023-03-01') -> str:
    """
    查詢在xx日期，某個xxxx股票號的技術形態

    Parameters:
        stock_code (str): 股票代碼
        acdate (str): 日期

    Returns:
        str: The query result
    """
    query = """
        select distinct  			t1.ACDATE,t1.code,t1.PATTERN_ID,t1.PATTERN_NAME 			from SRCIFF.TB_FTS_DAILYCHARTPATTERN t1 			where t1.acdate='{acdate}' 			and t1.confirmed_date<='{acdate}' 			and t1.expiry_date>'{acdate}' 			and t1.CODE='{stock_code}' 			and t1.pattern_ID in (5,6,7,8,13,14,26,27,30,31,32,34,35,36,41,42,43,44,45,46,50,51,60,61,62,63,70,71)
    """

    # Correct replacement using the actual parameter names in curly braces
    query = query.format(
        stock_code=str(stock_code) if stock_code is not None else "", acdate=str(acdate) if acdate is not None else ""
    )

    # Execute query
    return f"Executing query: {query}"



def query_get_prediction_by_date_and_stock(company_name: str = '中國銀行', acdate: str = '2023-03-01') -> str:
    """
    查詢在xx日期，某個xx股票(公司名)的預測市盈率

    Parameters:
        company_name (str): 股票代碼
        acdate (str): 日期

    Returns:
        str: The query result
    """
    query = """
        select 		t1.DATE,t1.code,1/nullifzero(t1.earning_yld_estimate) as estimated_PE 		,t3.STOCK_NAME,t3.CHI_NAME 		from SRCIFF.TB_FTS_DAILYCOMPINFO t1 		left join srciff.TB_DW46SK1 t3 on t3.acdate=t1.acdate and t3.stock_code=t1.code 		where t1.acdate='{acdate}' 		and t3.CHI_NAME like '{company_name}'
    """

    # Correct replacement using the actual parameter names in curly braces
    query = query.format(
        company_name=str(company_name) if company_name is not None else "", acdate=str(acdate) if acdate is not None else ""
    )

    # Execute query
    return f"Executing query: {query}"



def query_get_highest_yield_stocks_by_date(index_code: str = 'HSI', number: str = '5', acdate: str = '2023-03-01') -> str:
    """
    查詢在xx日期，xx指數成分股預測收益率最高的n只股票

    Parameters:
        index_code (str): 股票指數
        number (str): 指數中的成分股的數量
        acdate (str): 日期

    Returns:
        str: The query result
    """
    query = """
        select 		t1.DATE,t1.INDEX_CODE,t1.STOCK_CODE,t1.WEIGHT 		,t2.EARNING_YLD_ESTIMATE 		,t3.STOCK_NAME,t3.CHI_NAME 		from SRCIFF.TB_FTS_DAILYINDEXCONSTITUENT t1 		left join SRCIFF.TB_FTS_DAILYCOMPINFO t2 on t2.acdate=t1.acdate and t2.date=t1.date and t2.code=t1.stock_code 		left join srciff.TB_DW46SK1 t3 on t3.acdate=t1.acdate and t3.stock_code=t1.stock_code 		where t1.acdate='{acdate}' 		and t1.index_code='{index_code}' 		order by t2.EARNING_YLD_ESTIMATE desc limit '{number}'
    """

    # Correct replacement using the actual parameter names in curly braces
    query = query.format(
        index_code=str(index_code) if index_code is not None else "", number=str(number) if number is not None else "", acdate=str(acdate) if acdate is not None else ""
    )

    # Execute query
    return f"Executing query: {query}"



def query_find_stocks_by_date_industry_ratio(industry_code: str = 'BNK', index_code: str = 'HSI', acdate: str = '2023-03-01') -> str:
    """
    查詢在xx日期，xx行業在xx指數有多少隻股票，以及比重

    Parameters:
        industry_code (str): 行業對應代碼
        index_code (str): 股票指數代碼
        acdate (str): 日期

    Returns:
        str: The query result
    """
    query = """
        select  		t1.DATE,t1.INDEX_CODE 		,count(t1.STOCK_CODE) as number_of_stock 		,sum(t1.WEIGHT) as total_percent 		from SRCIFF.TB_FTS_DAILYINDEXCONSTITUENT t1 		left join SRCIFF.TB_FTS_DAILYCOMPINFO t2 on t2.acdate=t1.acdate and t2.date=t1.date and t2.code=t1.stock_code 		where t1.acdate='{acdate}' 		and t1.index_code='{index_code}' 		and (t2.industry='{industry_code}' 		or t2.industry='{industry_code}') 		group by 		t1.DATE,t1.INDEX_CODE
    """

    # Correct replacement using the actual parameter names in curly braces
    query = query.format(
        industry_code=str(industry_code) if industry_code is not None else "", index_code=str(index_code) if index_code is not None else "", acdate=str(acdate) if acdate is not None else ""
    )

    # Execute query
    return f"Executing query: {query}"



def query_worst_stock_performance_on_date(index_code: str = 'HSI', acdate: str = '2023-03-01') -> str:
    """
    查詢在xx日期，xxx指數表現最差的股票

    Parameters:
        index_code (str): 股票指數代碼
        acdate (str): 日期

    Returns:
        str: The query result
    """
    query = """
        select  		t1.DATE,t1.INDEX_CODE 		,t1.STOCK_CODE 		,t2.pe_chg_1d 		,t3.STOCK_NAME,t3.CHI_NAME 		from SRCIFF.TB_FTS_DAILYINDEXCONSTITUENT t1 		left join SRCIFF.TB_FTS_DAILYPRICEINFO t2 on t2.acdate=t1.acdate and t2.date=t1.date and t2.code=t1.stock_code 		left join srciff.TB_DW46SK1 t3 on t3.acdate=t1.acdate and t3.stock_code=t1.stock_code 		where t1.acdate= 		(select max(acdate) from SRCIFF.TB_FTS_DAILYINDEXCONSTITUENT where acdate <= current_date-7) 		and t1.index_code='{index_code}' 		order by t2.pe_chg_1d desc 		limit 1
    """

    # Correct replacement using the actual parameter names in curly braces
    query = query.format(
        index_code=str(index_code) if index_code is not None else "", acdate=str(acdate) if acdate is not None else ""
    )

    # Execute query
    return f"Executing query: {query}"



def query_find_min_hk_stock_by_date_and_industry(industry_code: str = 'BNK', acdate: str = '2023-03-01', number: str = '5') -> str:
    """
    查詢xx日期，xx行業PB/ratio最低的n只香港股票

    Parameters:
        industry_code (str): 行業對應代碼
        acdate (str): 日期
        number (str): 篩選股票數量

    Returns:
        str: The query result
    """
    query = """
        select  		t1.DATE,t1.CODE,t1.INDUSTRY 		,t3.STOCK_NAME,t3.CHI_NAME 		,t1.PB_RATIO 		from SRCIFF.TB_FTS_DAILYCOMPINFO t1 		left join srciff.TB_DW46SK1 t3 on t3.acdate=t1.acdate and t3.stock_code=t1.code 		where t1.acdate='{acdate}' 		and t1.industry='{industry_code}' 		and t1.PB_RATIO is not null 		order by t1.PB_RATIO asc 		limit '{number}'
    """

    # Correct replacement using the actual parameter names in curly braces
    query = query.format(
        industry_code=str(industry_code) if industry_code is not None else "", acdate=str(acdate) if acdate is not None else "", number=str(number) if number is not None else ""
    )

    # Execute query
    return f"Executing query: {query}"



def query_get_portfolio_weight_above(index_code: str = 'HSI', acdate: str = '2023-03-01', weight: str = '10') -> str:
    """
    查詢xx日期，在xx指數中比重超過n%的板塊

    Parameters:
        index_code (str): 股票指數代碼
        acdate (str): 日期
        weight (str): 權重

    Returns:
        str: The query result
    """
    query = """
        With ind_con as ( 		select  		t1.ACDATE,t1.INDEX_CODE,t1.STOCK_CODE,t1.WEIGHT 		,t2.INDUSTRY 		from SRCIFF.TB_FTS_DAILYINDEXCONSTITUENT t1 		left join SRCIFF.TB_FTS_DAILYCOMPINFO t2 on t2.acdate=t1.acdate and t2.date=t1.date and t2.code=t1.stock_code 		where t1.acdate='{acdate}' 		and t1.index_code='{index_code}' 		) 		select  		t10.ACDATE,t10.INDUSTRY 		,sum(t10.WEIGHT) as industry_weight 		from ind_con t10 		group by  		t10.ACDATE,t10.INDUSTRY 		having sum(t10.WEIGHT)>'{weight}' 		order by sum(t10.WEIGHT) desc
    """

    # Correct replacement using the actual parameter names in curly braces
    query = query.format(
        index_code=str(index_code) if index_code is not None else "", acdate=str(acdate) if acdate is not None else "", weight=str(weight) if weight is not None else ""
    )

    # Execute query
    return f"Executing query: {query}"



def query_get_time_range_change(start_date: str = '2023-01-01', end_date: str = '2023-03-31', index_code: str = 'HSI') -> str:
    """
    查詢xx時間段，xx指數的點數變化

    Parameters:
        start_date (str): 開始日期
        end_date (str): 結束日期
        index_code (str): 股票指數代碼

    Returns:
        str: The query result
    """
    query = """
        select  		t1.CODE, 		sum(change) as agg_change 		from SRCIFF.TB_FTS_DAILYINDEX t1 		where  		t1.CODE='{index_code}' 		and t1.ACDATE between '{start_date}' and '{end_date}' 		group by 		t1.CODE
    """

    # Correct replacement using the actual parameter names in curly braces
    query = query.format(
        start_date=str(start_date) if start_date is not None else "", end_date=str(end_date) if end_date is not None else "", index_code=str(index_code) if index_code is not None else ""
    )

    # Execute query
    return f"Executing query: {query}"



def query_get_return_rate_by_time_range_and_stock_id(start_date: str = '2023-01-01', end_date: str = '2023-03-31', stock_code: str = '5') -> str:
    """
    查詢xx時間段，xx股票號的回報率

    Parameters:
        start_date (str): 開始日期
        end_date (str): 結束日期
        stock_code (str): 股票代碼

    Returns:
        str: The query result
    """
    query = """
        With background_data as ( 		select t1.ACDATE,t1.DATE,t1.CODE,t1.SECURITY_TYPE,t1.CLOSE,t1.PER_CHG_1D,t1.TURNOVER 		,t1.COUNTER,t1.COMPANY_CODE,t1.CUR,t1.LIST_DATE,coalesce(t1.LIST_DATE,'1970-01-01') as LIST_DATE_DUMMY 		,ln(PER_CHG_1D/100+1) as change_1D_log,CASE when t1.CODE=t1.COUNTER then 1 else 2 end as stock_priority 		from SRCIFF.TB_FTS_DAILYPRICEINFO t1 		where acdate<='{end_date}' and acdate>='{start_date}' 		and t1.SECURITY_TYPE in ('STK','TRT','TRT_LIP','TRT_REI')-- can it be shown that it does not support this securities type 		and (t1.counter='{stock_code}' or t1.code='{stock_code}') 		order by t1.ACDATE,t1.CODE), 		last_trading_date_data as( 		select distinct t2.ACDATE,t2.COUNTER,t2.COMPANY_CODE,t2.CUR from background_data t2 		where t2.acdate=(select max(acdate) from SRCIFF.TB_FTS_DAILYPRICEINFO where acdate<='{end_date}' and 		acdate>'{end_date}'::date-7)), 		stock_selected as( 		select t5.* from 		(select t4.*,rank() over (partition by t4.COMPANY_CODE,t4.COUNTER,t4.ACDATE,t4.DATE order by t4.stock_priority) as ranking 		from ( 		--- to make sure no reuse RIC is selected by taking out same company_code 		select 		t3.* from background_data t3 inner join last_trading_date_data t4 on t4.company_code =t3.company_code and t4.counter=t3.counter 		) t4 		) t5 where t5.ranking=1) 		select 		t7.CODE_ORI,exp (change_cum_log)-1 as percent_chg 		from (select t6.COUNTER as CODE_ORI, sum(t6.change_1D_log) as change_cum_log from stock_selected t6 group by t6.COUNTER) t7
    """

    # Correct replacement using the actual parameter names in curly braces
    query = query.format(
        start_date=str(start_date) if start_date is not None else "", end_date=str(end_date) if end_date is not None else "", stock_code=str(stock_code) if stock_code is not None else ""
    )

    # Execute query
    return f"Executing query: {query}"



def query_get_stock_turnover_by_time(start_date: str = '2023-01-01', end_date: str = '2023-03-31', stock_code: str = '5') -> str:
    """
    查詢xx時間段，xx股票的turnover

    Parameters:
        start_date (str): 開始日期
        end_date (str): 結束日期
        stock_code (str): 股票代碼

    Returns:
        str: The query result
    """
    query = """
        With background_data as ( 		select 		t1.ACDATE,t1.DATE,t1.CODE,t1.SECURITY_TYPE,t1.CLOSE,t1.PER_CHG_1D,t1.TURNOVER 		,t1.COUNTER,t1.COMPANY_CODE,t1.CUR,t1.LIST_DATE,coalesce(t1.LIST_DATE,'1970-01-01') as LIST_DATE_DUMMY 		,ln(PER_CHG_1D/100+1) as change_1D_log,CASE when t1.CODE=t1.COUNTER then 1 else 2 end as stock_priority 		from SRCIFF.TB_FTS_DAILYPRICEINFO t1 		where acdate<='{end_date}' and acdate>='{start_date}' 		and t1.SECURITY_TYPE in ('STK','TRT','TRT_LIP','TRT_REI')-- can it be shown that it does not support this securities type 		and (t1.counter='{stock_code}' or t1.code='{stock_code}') 		order by t1.ACDATE,t1.CODE), 		last_trading_date_data as( 		select distinct 		t2.ACDATE,t2.COUNTER,t2.COMPANY_CODE,t2.CUR from background_data t2 		where t2.acdate=(select max(acdate) from SRCIFF.TB_FTS_DAILYPRICEINFO where acdate<='{end_date}' and 		acdate>'{end_date}'::date-7)), 		stock_selected as( 		select t5.* from 		(select t4.*,rank() over (partition by t4.COMPANY_CODE,t4.COUNTER,t4.ACDATE,t4.DATE order by t4.stock_priority) as ranking 		from 		( 		--- to make sure no reuse RIC is selected by taking out same company_code 		select 		t3.* from background_data t3 inner join last_trading_date_data t4 on t4.company_code =t3.company_code and t4.counter=t3.counter 		) t4 		)t5 --where t5.ranking=1 		select 		--t6.*
    """

    # Correct replacement using the actual parameter names in curly braces
    query = query.format(
        start_date=str(start_date) if start_date is not None else "", end_date=str(end_date) if end_date is not None else "", stock_code=str(stock_code) if stock_code is not None else ""
    )

    # Execute query
    return f"Executing query: {query}"



# Tool classes


class QueryGetHighestLowestPriceTool(BaseTool):
    """
    查詢在xx日期，美股xxxx股票號的最高價、最低價及收市價

    Example query: 
    
    Parameters:
        stock_code (str): 股票代碼
        acdate (str): 日期
    """

    name: str = "query_get_highest_lowest_price_tool"
    description: str = "查詢在xx日期，美股xxxx股票號的最高價、最低價及收市價"

    def _run(self, stock_code: str = '5', acdate: str = '2023-03-01') -> str:
        return query_get_highest_lowest_price(stock_code=stock_code, acdate=acdate)

    async def _arun(self, stock_code: str = '5', acdate: str = '2023-03-01') -> str:
        """Async version of _run."""
        return self._run(stock_code=stock_code, acdate=acdate)



class QueryGetStockTechnologyByDateTool(BaseTool):
    """
    查詢在xx日期，某個xxxx股票號的技術形態

    Example query: 
    
    Parameters:
        stock_code (str): 股票代碼
        acdate (str): 日期
    """

    name: str = "query_get_stock_technology_by_date_tool"
    description: str = "查詢在xx日期，某個xxxx股票號的技術形態"

    def _run(self, stock_code: str = '5', acdate: str = '2023-03-01') -> str:
        return query_get_stock_technology_by_date(stock_code=stock_code, acdate=acdate)

    async def _arun(self, stock_code: str = '5', acdate: str = '2023-03-01') -> str:
        """Async version of _run."""
        return self._run(stock_code=stock_code, acdate=acdate)



class QueryGetPredictionByDateAndStockTool(BaseTool):
    """
    查詢在xx日期，某個xx股票(公司名)的預測市盈率

    Example query: 
    
    Parameters:
        company_name (str): 股票代碼
        acdate (str): 日期
    """

    name: str = "query_get_prediction_by_date_and_stock_tool"
    description: str = "查詢在xx日期，某個xx股票(公司名)的預測市盈率"

    def _run(self, company_name: str = '中國銀行', acdate: str = '2023-03-01') -> str:
        return query_get_prediction_by_date_and_stock(company_name=company_name, acdate=acdate)

    async def _arun(self, company_name: str = '中國銀行', acdate: str = '2023-03-01') -> str:
        """Async version of _run."""
        return self._run(company_name=company_name, acdate=acdate)



class QueryGetHighestYieldStocksByDateTool(BaseTool):
    """
    查詢在xx日期，xx指數成分股預測收益率最高的n只股票

    Example query: 
    
    Parameters:
        index_code (str): 股票指數
        number (str): 指數中的成分股的數量
        acdate (str): 日期
    """

    name: str = "query_get_highest_yield_stocks_by_date_tool"
    description: str = "查詢在xx日期，xx指數成分股預測收益率最高的n只股票"

    def _run(self, index_code: str = 'HSI', number: str = '5', acdate: str = '2023-03-01') -> str:
        return query_get_highest_yield_stocks_by_date(index_code=index_code, number=number, acdate=acdate)

    async def _arun(self, index_code: str = 'HSI', number: str = '5', acdate: str = '2023-03-01') -> str:
        """Async version of _run."""
        return self._run(index_code=index_code, number=number, acdate=acdate)



class QueryFindStocksByDateIndustryRatioTool(BaseTool):
    """
    查詢在xx日期，xx行業在xx指數有多少隻股票，以及比重

    Example query: 
    
    Parameters:
        industry_code (str): 行業對應代碼
        index_code (str): 股票指數代碼
        acdate (str): 日期
    """

    name: str = "query_find_stocks_by_date_industry_ratio_tool"
    description: str = "查詢在xx日期，xx行業在xx指數有多少隻股票，以及比重"

    def _run(self, industry_code: str = 'BNK', index_code: str = 'HSI', acdate: str = '2023-03-01') -> str:
        return query_find_stocks_by_date_industry_ratio(industry_code=industry_code, index_code=index_code, acdate=acdate)

    async def _arun(self, industry_code: str = 'BNK', index_code: str = 'HSI', acdate: str = '2023-03-01') -> str:
        """Async version of _run."""
        return self._run(industry_code=industry_code, index_code=index_code, acdate=acdate)



class QueryWorstStockPerformanceOnDateTool(BaseTool):
    """
    查詢在xx日期，xxx指數表現最差的股票

    Example query: 
    
    Parameters:
        index_code (str): 股票指數代碼
        acdate (str): 日期
    """

    name: str = "query_worst_stock_performance_on_date_tool"
    description: str = "查詢在xx日期，xxx指數表現最差的股票"

    def _run(self, index_code: str = 'HSI', acdate: str = '2023-03-01') -> str:
        return query_worst_stock_performance_on_date(index_code=index_code, acdate=acdate)

    async def _arun(self, index_code: str = 'HSI', acdate: str = '2023-03-01') -> str:
        """Async version of _run."""
        return self._run(index_code=index_code, acdate=acdate)



class QueryFindMinHkStockByDateAndIndustryTool(BaseTool):
    """
    查詢xx日期，xx行業PB/ratio最低的n只香港股票

    Example query: 
    
    Parameters:
        industry_code (str): 行業對應代碼
        acdate (str): 日期
        number (str): 篩選股票數量
    """

    name: str = "query_find_min_hk_stock_by_date_and_industry_tool"
    description: str = "查詢xx日期，xx行業PB/ratio最低的n只香港股票"

    def _run(self, industry_code: str = 'BNK', acdate: str = '2023-03-01', number: str = '5') -> str:
        return query_find_min_hk_stock_by_date_and_industry(industry_code=industry_code, acdate=acdate, number=number)

    async def _arun(self, industry_code: str = 'BNK', acdate: str = '2023-03-01', number: str = '5') -> str:
        """Async version of _run."""
        return self._run(industry_code=industry_code, acdate=acdate, number=number)



class QueryGetPortfolioWeightAboveTool(BaseTool):
    """
    查詢xx日期，在xx指數中比重超過n%的板塊

    Example query: 
    
    Parameters:
        index_code (str): 股票指數代碼
        acdate (str): 日期
        weight (str): 權重
    """

    name: str = "query_get_portfolio_weight_above_tool"
    description: str = "查詢xx日期，在xx指數中比重超過n%的板塊"

    def _run(self, index_code: str = 'HSI', acdate: str = '2023-03-01', weight: str = '10') -> str:
        return query_get_portfolio_weight_above(index_code=index_code, acdate=acdate, weight=weight)

    async def _arun(self, index_code: str = 'HSI', acdate: str = '2023-03-01', weight: str = '10') -> str:
        """Async version of _run."""
        return self._run(index_code=index_code, acdate=acdate, weight=weight)



class QueryGetTimeRangeChangeTool(BaseTool):
    """
    查詢xx時間段，xx指數的點數變化

    Example query: 
    
    Parameters:
        start_date (str): 開始日期
        end_date (str): 結束日期
        index_code (str): 股票指數代碼
    """

    name: str = "query_get_time_range_change_tool"
    description: str = "查詢xx時間段，xx指數的點數變化"

    def _run(self, start_date: str = '2023-01-01', end_date: str = '2023-03-31', index_code: str = 'HSI') -> str:
        return query_get_time_range_change(start_date=start_date, end_date=end_date, index_code=index_code)

    async def _arun(self, start_date: str = '2023-01-01', end_date: str = '2023-03-31', index_code: str = 'HSI') -> str:
        """Async version of _run."""
        return self._run(start_date=start_date, end_date=end_date, index_code=index_code)



class QueryGetReturnRateByTimeRangeAndStockIdTool(BaseTool):
    """
    查詢xx時間段，xx股票號的回報率

    Example query: 
    
    Parameters:
        start_date (str): 開始日期
        end_date (str): 結束日期
        stock_code (str): 股票代碼
    """

    name: str = "query_get_return_rate_by_time_range_and_stock_id_tool"
    description: str = "查詢xx時間段，xx股票號的回報率"

    def _run(self, start_date: str = '2023-01-01', end_date: str = '2023-03-31', stock_code: str = '5') -> str:
        return query_get_return_rate_by_time_range_and_stock_id(start_date=start_date, end_date=end_date, stock_code=stock_code)

    async def _arun(self, start_date: str = '2023-01-01', end_date: str = '2023-03-31', stock_code: str = '5') -> str:
        """Async version of _run."""
        return self._run(start_date=start_date, end_date=end_date, stock_code=stock_code)



class QueryGetStockTurnoverByTimeTool(BaseTool):
    """
    查詢xx時間段，xx股票的turnover

    Example query: 
    
    Parameters:
        start_date (str): 開始日期
        end_date (str): 結束日期
        stock_code (str): 股票代碼
    """

    name: str = "query_get_stock_turnover_by_time_tool"
    description: str = "查詢xx時間段，xx股票的turnover"

    def _run(self, start_date: str = '2023-01-01', end_date: str = '2023-03-31', stock_code: str = '5') -> str:
        return query_get_stock_turnover_by_time(start_date=start_date, end_date=end_date, stock_code=stock_code)

    async def _arun(self, start_date: str = '2023-01-01', end_date: str = '2023-03-31', stock_code: str = '5') -> str:
        """Async version of _run."""
        return self._run(start_date=start_date, end_date=end_date, stock_code=stock_code)



def get_financial_query_tools() -> List[BaseTool]:
    """Return a list of all financial query tools."""
    return [
        QueryGetHighestLowestPriceTool(),
        QueryGetStockTechnologyByDateTool(),
        QueryGetPredictionByDateAndStockTool(),
        QueryGetHighestYieldStocksByDateTool(),
        QueryFindStocksByDateIndustryRatioTool(),
        QueryWorstStockPerformanceOnDateTool(),
        QueryFindMinHkStockByDateAndIndustryTool(),
        QueryGetPortfolioWeightAboveTool(),
        QueryGetTimeRangeChangeTool(),
        QueryGetReturnRateByTimeRangeAndStockIdTool(),
        QueryGetStockTurnoverByTimeTool()
    ]
