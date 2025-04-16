"""
Configuration settings for financial query tools.
"""

# Ollama API configuration
OLLAMA_API_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3.2"

# File paths
# LIST_QUESTION_PATH = "list_question.py"
QUERY_TOOLS_OUTPUT_PATH = "query_tools.py"

# File paths
FINANCIAL_QUERIES_CSV = "financial.csv"  # Path to the CSV file with query definitions

# Parameter descriptions
PARAMETER_DESCRIPTIONS = {
    "acdate": "The accounting date in YYYY-MM-DD format",
    "stock_code": "The stock code or ticker symbol",
    "company_name": "The company name in Chinese",
    "index_code": "The index code (e.g., HSI for Hang Seng Index)",
    "industry_code": "The industry code (e.g., BNK for banking, INS for insurance)",
    "number": "The number of stocks to return",
    "weight": "The weight threshold percentage",
    "start_date": "The start date in YYYY-MM-DD format",
    "end_date": "The end date in YYYY-MM-DD format"
}

# Default parameter values
DEFAULT_PARAMETER_VALUES = {
    "acdate": "2023-03-01",
    "start_date": "2023-01-01",
    "end_date": "2023-03-31",
    "stock_code": "5",
    "company_name": "中國銀行",
    "index_code": "HSI",
    "industry_code": "BNK",
    "number": 5,
    "weight": 10
}

# Translation prompt
TRANSLATION_PROMPT = """Translate the following Chinese financial term to a VERY SHORT English function name (1-2 words maximum, lowercase with underscore if needed). 
Be extremely concise - the shorter the better.
Don't include 'query' in your translation. 
Only provide the translated term, nothing else:

'{text}'"""

# Regular expression patterns
QUESTION_PATTERN = r'{\s*"question":\s*"([^"]+)",\s*"intension":\s*"([^"]+)",\s*"input_parameter":\s*{([^}]+)},\s*"sql":\s*"""([^"]+)"""'
PARAM_PATTERN = r'"([^"]+)":\s*xxx'
