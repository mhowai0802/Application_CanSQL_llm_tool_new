"""
Script to generate query_tools.py from a CSV file containing financial queries.
Uses Ollama to translate Chinese intensions to English function names.
"""

import re
import json
import requests
import csv
import os
from typing import List, Dict, Any, Set, Optional, Tuple
import time
import sys

# Import configuration from config.py
from config import OLLAMA_API_URL, OLLAMA_MODEL, FINANCIAL_QUERIES_CSV
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
    """Load questions from a CSV file."""
    questions = []

    try:
        with open(csv_path, mode='r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                # Parse input parameters and descriptions
                input_params = {}
                param_descriptions = {}

                if row.get('input_parameter') and row.get('input_description'):
                    params = [p.strip() for p in row['input_parameter'].split(',') if p.strip()]
                    descriptions = [d.strip() for d in row['input_description'].split(',') if d.strip()]

                    # Match parameters with descriptions (up to the number of available descriptions)
                    for i, param in enumerate(params):
                        input_params[param] = 'xxx'  # Default value
                        if i < len(descriptions):
                            param_descriptions[param] = descriptions[i]

                question = {
                    'question': row.get('question', ''),
                    'intension': row.get('intention', ''),  # Note: CSV column is "intention" not "intension"
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


def generate_function_definition(question: Dict[str, Any], function_name: str) -> str:
    """Generate a Python function definition for a query."""
    intension = question['intension']

    # Get parameters from input_parameter
    params = list(question.get('input_parameter', {}).keys())

    # Default parameters
    param_defaults = {
        'acdate': "'2023-03-01'",
        'stock_code': "'5'",
        'company_name': "'中國銀行'",
        'index_code': "'HSI'",
        'industry_code': "'BNK'",
        'number': "'5'",
        'weight': "'10'",
        'start_date': "'2023-01-01'",
        'end_date': "'2023-03-31'",
    }

    # Generate parameter list
    param_list = []
    for param in params:
        default = param_defaults.get(param, "None")
        param_list.append(f"{param}: str = {default}")

    # Format SQL query with pretty formatting
    raw_sql = question.get('sql', '-- No query provided')
    formatted_sql = format_sql_query_pretty(raw_sql)

    # Check if it's a numeric parameter that shouldn't be quoted in SQL
    numeric_params = ['number', 'weight']
    formatted_params = []
    for param in params:
        if param in numeric_params:
            formatted_params.append(f"{param}=str({param}) if {param} is not None else \"\"")
        else:
            formatted_params.append(f"{param}=str({param}) if {param} is not None else \"\"")

    # Parameter descriptions - use custom descriptions from CSV if available
    param_descriptions = []
    for param in params:
        desc = question.get('param_descriptions', {}).get(param, get_default_param_description(param))
        param_descriptions.append(f"{param} (str): {desc}")

    # Join param descriptions with proper indentation
    param_desc_str = "\n        ".join(param_descriptions)

    # Function template with nicely formatted SQL
    function_code = f"""
def {function_name}({', '.join(param_list)}) -> str:
    \"\"\"
    {intension}

    Parameters:
        {param_desc_str}

    Returns:
        str: The query result
    \"\"\"
    query = \"\"\"
{formatted_sql}
    \"\"\"

    # Correct replacement using the actual parameter names in curly braces
    query = query.format(
        {', '.join(formatted_params)}
    )

    # Execute query
    return f"Executing query: {{query}}"
"""
    return function_code


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
            parts.append(' ' * 8 + token)  # 8 spaces for indentation

        # Check for join keywords
        elif upper_token in join_keywords or any(upper_token == j for j in join_keywords):
            if parts and parts[-1] != '\n':
                parts.append('\n')
            parts.append(' ' * 12 + token)  # 12 spaces for join indentation

        # Check for condition keywords
        elif upper_token in condition_keywords:
            if parts and parts[-1] != '\n':
                parts.append('\n')
            parts.append(' ' * 16 + token)  # 16 spaces for condition indentation

        # Handle opening parenthesis for subqueries
        elif token == '(' and i + 2 < len(tokens) and tokens[i + 2].upper() == 'SELECT':
            parts.append(token)
            parts.append('\n')
            parts.append(' ' * 12)  # Extra indentation for subqueries

        # Handle commas - add newlines after commas when not inside parentheses
        elif token == ',':
            parts.append(token)
            # Check if we're not inside a function call
            if not is_inside_function(sql, ''.join(parts)):
                parts.append('\n')
                parts.append(' ' * 12)  # Indent for comma-separated items

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
                lines[i] = ' ' * 8 + lines[i]
            break

    formatted_sql = '\n'.join(lines)

    # Final adjustment to ensure correct indentation
    if not formatted_sql.startswith(' ' * 8):
        formatted_sql = ' ' * 8 + formatted_sql

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


def format_select_items(select_clause):
    """Format the items in a SELECT clause for better readability.
    Simplified version without regex lookbehinds.
    """
    # Extract the part between SELECT and FROM
    match = re.search(r'SELECT\s+(.*?)\s+FROM', select_clause, re.IGNORECASE | re.DOTALL)
    if not match:
        return select_clause

    items = match.group(1)
    # Use a simple comma-based split (this won't handle function calls with commas perfectly)
    simple_items = items.split(',')

    if len(simple_items) <= 1:
        return select_clause

    # Format the items with indentation
    formatted_items = []
    for item in simple_items:
        formatted_items.append(' ' * 12 + item.strip())

    # Reconstruct the SELECT clause
    result = "SELECT\n" + ',\n'.join(formatted_items) + "\nFROM"

    # Replace the original SELECT clause
    return select_clause.replace(match.group(0), result)


def format_select_expressions(select_clause):
    """Format a SELECT clause with aligned columns for better readability."""
    # Don't process if it's a simple SELECT *
    if re.search(r'SELECT\s+\*\s+FROM', select_clause, re.IGNORECASE):
        return select_clause

    # Extract the expressions between SELECT and FROM
    match = re.search(r'SELECT\s+(.+?)\s+FROM', select_clause, re.IGNORECASE | re.DOTALL)
    if not match:
        return select_clause

    expressions = match.group(1)

    # Split expressions by commas but preserve function commas
    # This is a simplified approach and may need refinement for complex nested expressions
    formatted_expressions = []
    current_expr = ""
    paren_level = 0

    for char in expressions:
        if char == '(':
            paren_level += 1
            current_expr += char
        elif char == ')':
            paren_level -= 1
            current_expr += char
        elif char == ',' and paren_level == 0:
            formatted_expressions.append(current_expr.strip())
            current_expr = ""
        else:
            current_expr += char

    if current_expr.strip():
        formatted_expressions.append(current_expr.strip())

    # Format the expressions with proper indentation
    if len(formatted_expressions) > 1:
        columns = "\n            " + ",\n            ".join(formatted_expressions)
        return f"SELECT{columns}\n        FROM"
    else:
        return select_clause

def generate_tool_class(question: Dict[str, Any], function_name: str) -> str:
    """Generate a tool class for a query function."""
    intension = question['intension']
    example_question = question.get('question', '')
    class_name = f"{''.join(word.capitalize() for word in function_name.split('_'))}Tool"

    # Get parameters from input_parameter
    params = list(question.get('input_parameter', {}).keys())

    # Default parameters
    param_defaults = {
        'acdate': "'2023-03-01'",
        'stock_code': "'5'",
        'company_name': "'中國銀行'",
        'index_code': "'HSI'",
        'industry_code': "'BNK'",
        'number': "'5'",
        'weight': "'10'",
        'start_date': "'2023-01-01'",
        'end_date': "'2023-03-31'",
    }

    # Generate parameter list for _run method
    param_list = []
    for param in params:
        default = param_defaults.get(param, "None")
        param_list.append(f"{param}: str = {default}")

    # Parameter descriptions for docstring - use custom descriptions if available
    # Format with each parameter on a new line with proper indentation
    param_descriptions = []
    for param in params:
        desc = question.get('param_descriptions', {}).get(param, get_default_param_description(param))
        param_descriptions.append(f"{param} (str): {desc}")

    # Join param descriptions with proper indentation
    param_desc_str = "\n        ".join(param_descriptions)

    # Tool class template with improved formatting
    tool_code = f"""
class {class_name}(BaseTool):
    \"\"\"
    {intension}

    Example query: {example_question}
    
    Parameters:
        {param_desc_str}
    \"\"\"

    name: str = "{function_name}_tool"
    description: str = "{intension}"

    def _run(self, {', '.join(param_list)}) -> str:
        return {function_name}({', '.join([f"{param}={param}" for param in params])})

    async def _arun(self, {', '.join(param_list)}) -> str:
        \"\"\"Async version of _run.\"\"\"
        return self._run({', '.join([f"{param}={param}" for param in params])})
"""
    return tool_code

def generate_tools_list(tool_classes: List[str]) -> str:
    """Generate code to create a list of all tool instances."""
    tool_names = [re.search(r'class (\w+)\(BaseTool\)', tool_class).group(1) for tool_class in tool_classes]

    tools_list_code = """
def get_financial_query_tools() -> List[BaseTool]:
    \"\"\"Return a list of all financial query tools.\"\"\"
    return [
        {tools}
    ]
""".format(tools=',\n        '.join([f"{name}()" for name in tool_names]))

    return tools_list_code

def generate_param_descriptions_dict(questions: List[Dict[str, Any]]) -> str:
    """Generate a dictionary of all parameter descriptions found in the CSV."""
    all_descriptions = {}

    # First collect custom descriptions from CSV
    for question in questions:
        if 'param_descriptions' in question:
            for param, desc in question['param_descriptions'].items():
                if param not in all_descriptions:
                    all_descriptions[param] = desc

    # Then add default descriptions for common parameters if not already described
    default_descriptions = {
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

    for param, desc in default_descriptions.items():
        if param not in all_descriptions:
            all_descriptions[param] = desc

    # Format as Python dictionary string
    dict_str = "{\n"
    for param, desc in sorted(all_descriptions.items()):
        dict_str += f'    "{param}": "{desc}",\n'
    dict_str += "}"

    return dict_str



def generate_query_tools_file(questions: List[Dict[str, Any]], output_file: str = "query_tools.py") -> None:
    """Generate the query_tools.py file."""
    # Create parameter descriptions dictionary from all questions
    param_descriptions_dict = generate_param_descriptions_dict(questions)

    # Header with info about the source CSV
    header = f"""\"\"\"
Query tools generated from {FINANCIAL_QUERIES_CSV} for financial data queries.
Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}
\"\"\"

from typing import Optional, Dict, Any, List, Union
from langchain.tools import BaseTool

# Parameter descriptions collected from CSV and defaults
PARAMETER_DESCRIPTIONS = {param_descriptions_dict}
"""

    # Generate functions
    functions = []
    tool_classes = []
    function_names = []

    # Test Ollama API connection before starting
    try:
        url = f"{OLLAMA_API_URL.rstrip('/')}/api/tags"
        response = requests.get(url)
        if response.status_code == 200:
            models = [m.get('name', 'unknown') for m in response.json().get('models', [])]
            print(f"Ollama API connection successful! Available models: {', '.join(models)}")
            if OLLAMA_MODEL not in models:
                print(f"Warning: Model '{OLLAMA_MODEL}' not found in available models. Translation may fail.")
        else:
            print(f"Ollama API connection failed with status {response.status_code}. Will use fallback function naming.")
    except Exception as e:
        print(f"Ollama API connection test failed: {str(e)}. Will use fallback function naming.")

    for question in questions:
        # Skip if no intension
        if not question.get('intension'):
            continue

        # Fix the typo in the turnover query
        if "turnover" in question.get('intension', '').lower():
            if "acdate>='%{start_date}'" in question.get('sql', ''):
                question['sql'] = question['sql'].replace("acdate>='%{start_date}'", "acdate>='{start_date}'")

        # Generate function name using the Ollama model
        print(f"Translating: {question['intension']}")
        base_name = translate_to_english(question['intension'])
        # Add query_ prefix to all function names
        function_name = f"query_{base_name}"

        # Ensure unique function names
        if function_name in function_names:
            count = 1
            while f"{function_name}_{count}" in function_names:
                count += 1
            function_name = f"{function_name}_{count}"

        function_names.append(function_name)

        # Generate function code and tool class
        function_code = generate_function_definition(question, function_name)
        functions.append(function_code)

        tool_class = generate_tool_class(question, function_name)
        tool_classes.append(tool_class)

        # Add a small delay to avoid overwhelming the API
        time.sleep(0.5)

    # Generate tools list
    tools_list = generate_tools_list(tool_classes)

    # Combine all parts
    file_content = header + '\n\n' + '\n\n'.join(functions) + '\n\n\n# Tool classes\n\n' + '\n\n'.join(tool_classes) + '\n\n' + tools_list

    # Write to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(file_content)

    print(f"Generated {output_file} with {len(functions)} functions and {len(tool_classes)} tool classes.")




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