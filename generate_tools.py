"""
Script to generate query_tools.py from a CSV file containing financial queries.
Uses Ollama to translate Chinese intensions to English function names.
"""

import re
import requests
import csv
import os
from typing import List, Dict, Any, Tuple
import time
import sys

# Import configuration from config.py
from backend.config import OLLAMA_API_URL, OLLAMA_MODEL, FINANCIAL_QUERIES_CSV
print(f"Using configuration from config.py:")
print(f"  - OLLAMA_API_URL: {OLLAMA_API_URL}")
print(f"  - OLLAMA_MODEL: {OLLAMA_MODEL}")
print(f"  - FINANCIAL_QUERIES_CSV: {FINANCIAL_QUERIES_CSV}")


def translate_to_english(text: str) -> str:
    """
    Translate Chinese text to a short, precise English function name using Ollama model.
    """
    system_prompt = """
    As a translator, convert the Chinese database query intention into a short, precise English function name.
    Use snake_case format (lowercase with underscores).
    Keep it brief but descriptive - ideally 2-4 words.
    Do NOT add 'query_' prefix or any other prefixes.
    ONLY return the function name, nothing else.
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Translate this database query intention to a short, precise English function name: {text}"}
    ]

    try:
        url = f"{OLLAMA_API_URL.rstrip('/')}/api/chat"
        payload = {
            "model": OLLAMA_MODEL,
            "messages": messages,
            "stream": False
        }

        print(f"Sending translation request to: {url}")
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            result = response.json()
            translated_name = result["message"]["content"].strip().lower()
            # Clean up: ensure snake_case format and remove any quotes or punctuation
            translated_name = re.sub(r'[^a-z0-9_\s]', '', translated_name)
            translated_name = re.sub(r'\s+', '_', translated_name)

            print(f"Translated: '{text}' → '{translated_name}'")

            # If translation failed or returned something too short, use fallback
            if len(translated_name) < 3:
                return generate_snake_case_function_name(text)

            return translated_name
        else:
            print(f"Translation API error: {response.status_code} - {response.text}")
            return generate_snake_case_function_name(text)
    except Exception as e:
        print(f"Translation failed: {str(e)}")
        return generate_snake_case_function_name(text)


def load_questions_from_csv(csv_path: str) -> List[Dict[str, Any]]:
    """Load questions from a CSV file with proper character encoding handling."""
    questions = []

    try:
        with open(csv_path, mode='r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                # Parse input parameters and descriptions
                input_params = {}
                param_descriptions = {}

                # Check if the columns exist and have values
                if row.get('input_parameter') and row.get('input_description'):
                    # Split by comma and clean up whitespace
                    params = [p.strip() for p in row['input_parameter'].split(',') if p.strip()]
                    descriptions = [d.strip() for d in row['input_description'].split(',') if d.strip()]

                    # Common parameter names in English
                    standard_param_names = [
                        'stock_code', 'acdate', 'company_name', 'index_code',
                        'industry_code', 'number', 'weight', 'start_date', 'end_date'
                    ]

                    # Parameter name mapping (Chinese to English)
                    param_name_mapping = {
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

                    # Match parameters with descriptions
                    for i, param in enumerate(params):
                        # Try to map Chinese parameter names to English
                        param_name = param_name_mapping.get(param, param)

                        # If still not in standard names, try to guess
                        if param_name not in standard_param_names:
                            # Check if it's already in English
                            if param in standard_param_names:
                                param_name = param
                            else:
                                # Use the original name as fallback
                                param_name = param

                        # Add to input parameters with standard names
                        input_params[param_name] = 'xxx'  # Default value

                        # Add description if available
                        if i < len(descriptions):
                            param_descriptions[param_name] = descriptions[i]

                # Build question dictionary with correct column names
                question = {
                    'question': row.get('question', ''),
                    'intension': row.get('intention', '') or row.get('intension', ''),  # Try both spellings
                    'input_parameter': input_params,
                    'param_descriptions': param_descriptions,
                    'sql': row.get('sql', '')
                }
                questions.append(question)

        print(f"Loaded {len(questions)} questions from {csv_path}")
        return questions

    except Exception as e:
        print(f"Error loading CSV file {csv_path}: {str(e)}")
        return []

def generate_snake_case_function_name(intension: str) -> str:
    """Generate a snake_case function name from an intension (fallback method)."""
    # Remove special characters, convert to lowercase, replace spaces with underscores
    name = re.sub(r'[^a-zA-Z0-9\s]', '', intension.lower())
    name = re.sub(r'\s+', '_', name)
    # Return just the base name without prefix (prefix added later)
    return name

def get_default_param_description(param: str) -> str:
    """Get default description for a parameter."""
    descriptions = {
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
    return descriptions.get(param, "Parameter description not available")

def format_sql_query_pretty(sql: str) -> str:
    """Format SQL query in a very pretty, readable way with clear structure."""
    # Normalize whitespace first
    sql = sql.strip()
    sql = re.sub(r'\s+', ' ', sql)

    # Define major and minor SQL keywords
    major_keywords = [
        'SELECT', 'FROM', 'WHERE', 'GROUP BY', 'ORDER BY', 'HAVING',
        'LIMIT', 'OFFSET', 'WITH'
    ]

    join_keywords = [
        'JOIN', 'INNER JOIN', 'LEFT JOIN', 'RIGHT JOIN', 'FULL JOIN',
        'CROSS JOIN', 'NATURAL JOIN'
    ]

    condition_keywords = ['AND', 'OR']

    # Add proper newlines and indentation - avoid look-behind patterns
    parts = []
    tokens = re.split(r'(\s+)', sql)
    i = 0

    while i < len(tokens):
        token = tokens[i]
        upper_token = token.upper()

        # Check for major keywords
        if upper_token in major_keywords:
            if parts and parts[-1] != '\n':
                parts.append('\n')
            parts.append(' ' * 2 + token)  # 8 spaces for indentation

        # Check for join keywords
        elif upper_token in join_keywords or any(upper_token == j for j in join_keywords):
            if parts and parts[-1] != '\n':
                parts.append('\n')
            parts.append(' ' * 2 + token)  # 12 spaces for join indentation

        # Check for condition keywords
        elif upper_token in condition_keywords:
            if parts and parts[-1] != '\n':
                parts.append('\n')
            parts.append(' ' * 2 + token)  # 16 spaces for condition indentation

        # Handle opening parenthesis for subqueries
        elif token == '(' and i + 2 < len(tokens) and tokens[i + 2].upper() == 'SELECT':
            parts.append(token)
            parts.append('\n')
            parts.append(' ' * 2)  # Extra indentation for subqueries

        # Handle commas - add newlines after commas when not inside parentheses
        elif token == ',':
            parts.append(token)
            # Check if we're not inside a function call
            if not is_inside_function(sql, ''.join(parts)):
                parts.append('\n')
                parts.append(' ' * 2)  # Indent for comma-separated items

        # Default case - just add the token
        else:
            parts.append(token)

        i += 1

    # Combine all parts
    formatted_sql = ''.join(parts)

    # Final cleanup
    # Fix any extraneous whitespace
    formatted_sql = re.sub(r'\s+$', '', formatted_sql, flags=re.MULTILINE)
    formatted_sql = re.sub(r'\n\s*\n', '\n', formatted_sql)

    # Ensure first line is properly indented
    lines = formatted_sql.split('\n')
    for i in range(len(lines)):
        if lines[i].strip():
            if not re.match(r'^\s+', lines[i]):
                lines[i] = ' ' * 2 + lines[i]
            break

    formatted_sql = '\n'.join(lines)

    # Final adjustment to ensure correct indentation
    if not formatted_sql.startswith(' ' * 2):
        formatted_sql = ' ' * 2 + formatted_sql

    return formatted_sql


def is_inside_function(full_sql, processed_so_far):
    """Check if the current position is inside a function call by counting parentheses."""
    stack = []
    for char in processed_so_far:
        if char == '(':
            stack.append(char)
        elif char == ')' and stack:
            stack.pop()
    return len(stack) > 0

def generate_pydantic_schema(function_name: str, params: List[str], param_descriptions: Dict[str, str]) -> Tuple[str, str]:
    """Generate Pydantic model schema for the tool's arguments."""
    # Convert function name to PascalCase for the class name
    class_name = ''.join(word.capitalize() for word in function_name.split('_')) + 'Input'

    # Build the class attributes
    fields = []
    for param in params:
        # Get description for this parameter
        description = param_descriptions.get(param, get_default_param_description(param))

        # Determine field type based on parameter name
        field_type = "int" if param in ['number', 'weight'] else "str"

        # Add Field with description
        fields.append(f"    {param}: {field_type} = Field(description=\"{description}\")")

    # Build the complete class
    schema_code = f"""
class {class_name}(BaseModel):
{chr(10).join(fields)}
"""
    return schema_code, class_name

def generate_pydantic_tool_function(question: Dict[str, Any], function_name: str) -> str:
    """Generate a Pydantic-based tool function."""
    intension = question['intension']
    params = list(question.get('input_parameter', {}).keys())
    param_descriptions = question.get('param_descriptions', {})

    # Generate Pydantic schema
    schema_code, schema_class_name = generate_pydantic_schema(function_name, params, param_descriptions)

    # Get param list for the function
    param_lines = []
    for i, param in enumerate(params):
        param_type = "int" if param in ['number', 'weight'] else "str"
        # Add comma after each parameter except the last one
        comma = "," if i < len(params) - 1 else ""
        param_lines.append(f"    {param}: {param_type}{comma}")

    # Format SQL query with pretty formatting
    raw_sql = question.get('sql', '-- No query provided')
    formatted_sql = format_sql_query_pretty(raw_sql)

    # Generate the function code
    function_code = f"""{schema_code}

@tool("{function_name}", args_schema={schema_class_name}, return_direct=True)
def {function_name}(
{chr(10).join(param_lines)}
):
    \"\"\"
    {intension}
    
    Returns:
        str: The query result
    \"\"\"
    query = \"\"\"
{formatted_sql}
    \"\"\"

    # Format parameters in the query
    query = query.format(
        {', '.join([f'{param}=str({param})' for param in params])}
    )

    # Execute query (placeholder for actual execution)
    return f"Executing query: {{query}}"
"""

    return function_code

def generate_query_tools_file(questions: List[Dict[str, Any]], output_file: str = "backend/query_tools.py") -> None:
    """Generate the query_tools.py file with Pydantic-based tool functions."""
    # Header
    header = f"""\"\"\"
Query tools generated from {FINANCIAL_QUERIES_CSV} for financial data queries.
Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}
\"\"\"

from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
from langchain.tools import tool

"""

    tool_functions = []

    for question in questions:
        # Skip if no intension
        if not question.get('intension'):
            continue

        # Generate function name using the Ollama model
        print(f"Translating: {question['intension']}")
        base_name = translate_to_english(question['intension'])

        # Ensure unique function names
        if base_name in [func.split('\n@tool("')[1].split('"')[0] for func in tool_functions if '\n@tool("' in func]:
            count = 1
            while f"{base_name}_{count}" in [func.split('\n@tool("')[1].split('"')[0] for func in tool_functions if '\n@tool("' in func]:
                count += 1
            base_name = f"{base_name}_{count}"

        # Generate the Pydantic tool function
        tool_function = generate_pydantic_tool_function(question, base_name)
        tool_functions.append(tool_function)

        # Add a small delay to avoid overwhelming the API
        time.sleep(0.5)

    # Combine all parts
    file_content = header + '\n\n'.join(tool_functions)

    # Write to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(file_content)

    print(f"Generated {output_file} with {len(tool_functions)} Pydantic-based tool functions.")


if __name__ == "__main__":
    # First, check if CSV path is provided via command line (highest priority)
    csv_path = FINANCIAL_QUERIES_CSV  # Default from config.py

    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
        print(f"Using CSV file from command line: {csv_path}")

    if not os.path.exists(csv_path):
        print(f"Error: CSV file '{csv_path}' not found. Please create this file or specify a different path.")
        sys.exit(1)

    questions = load_questions_from_csv(csv_path)
    if not questions:
        print("No questions loaded from CSV. Exiting.")
        sys.exit(1)

    generate_query_tools_file(questions)