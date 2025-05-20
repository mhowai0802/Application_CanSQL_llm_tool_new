"""
Configuration for generate_tools.py
"""

# Ollama configuration
OLLAMA_API_URL = "http://localhost:11434"  # Default value, can be overridden in config.py
OLLAMA_MODEL = "llama3"  # Default value, can be overridden in config.py

# File paths
FINANCIAL_QUERIES_CSV = "financial_queries.csv"  # Default value, can be overridden in config.py

# Common parameter names in English
STANDARD_PARAM_NAMES = [
    'stock_code', 'acdate', 'company_name', 'index_code',
    'industry_code', 'number', 'weight', 'start_date', 'end_date'
]

# Parameter name mapping (Chinese to English)
PARAM_NAME_MAPPING = {
    '股票代號': 'stock_code',
    '股票號碼': 'stock_code',
    '股票代码': 'stock_code',
    '股票編號': 'stock_code',
    '股票編碼': 'stock_code',
    '股號': 'stock_code',
    '日期': 'acdate',
    '時間': 'acdate',
    '時段': 'acdate',
    '當日': 'acdate',
    '當天': 'acdate',
    '收市日': 'acdate',
    '公司名稱': 'company_name',
    '公司': 'company_name',
    '企業名稱': 'company_name',
    '企業': 'company_name',
    '股票名稱': 'company_name',
    '指數代號': 'index_code',
    '指數編號': 'index_code',
    '指數代码': 'index_code',
    '指數': 'index_code',
    '行業代號': 'industry_code',
    '行業編號': 'industry_code',
    '行業類別': 'industry_code',
    '行業': 'industry_code',
    '數量': 'number',
    '數字': 'number',
    '個數': 'number',
    '數目': 'number',
    '數值': 'number',
    '權重': 'weight',
    '比重': 'weight',
    '重量': 'weight',
    '開始日期': 'start_date',
    '起始日期': 'start_date',
    '開始時間': 'start_date',
    '結束日期': 'end_date',
    '終止日期': 'end_date',
    '結束時間': 'end_date'
}

# Default parameter descriptions
DEFAULT_PARAM_DESCRIPTIONS = {
    'acdate': "The accounting date in YYYY-MM-DD format",
    'stock_code': "The stock code or ticker symbol",
    'company_name': "The company name in Chinese",
    'index_code': "The index code (e.g., HSI for Hang Seng Index)",
    'industry_code': "The industry code (e.g., BNK for banking, INS for insurance)",
    'number': "The number of stocks to return",
    'weight': "The weight threshold percentage",
    'start_date': "The start date in YYYY-MM-DD format",
    'end_date': "The end date in YYYY-MM-DD format",
}