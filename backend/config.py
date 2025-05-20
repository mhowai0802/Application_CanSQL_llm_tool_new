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


# Translation prompt
TRANSLATION_PROMPT = """Translate the following Chinese financial term to a VERY SHORT English function name (1-2 words maximum, lowercase with underscore if needed). 
Be extremely concise - the shorter the better.
Don't include 'query' in your translation. 
Only provide the translated term, nothing else:

'{text}'"""

# Regular expression patterns
QUESTION_PATTERN = r'{\s*"question":\s*"([^"]+)",\s*"intension":\s*"([^"]+)",\s*"input_parameter":\s*{([^}]+)},\s*"sql":\s*"""([^"]+)"""'
PARAM_PATTERN = r'"([^"]+)":\s*xxx'
