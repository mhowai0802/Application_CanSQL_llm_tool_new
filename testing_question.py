"""
Test script for the financial query LLM router.
Tests a list of predefined questions and outputs in UnifiedGenSQLResponse format.
"""
import json
import traceback
import uuid
from typing import Dict, Any

# Import the necessary functions from main.py
try:
    from main import get_ollama_response, chat_with_llm, execute_tool, parse_llm_response
    from backend.query_tools import (
        get_highest_lowest_close_prices_by_date_stock_number,
        get_technical_type_by_date_stock,
        get_stock_premium,
        get_top_stocks_by_date_and_weight,
        get_stock_holding_by_date_industry_ratio,
        worst_perform_stock_by_date,
        find_lowest_pb_hk_stock,
        find_weight_by_date,
        get_price_range_change,
        search_time_range_stock_report,
        get_stock_turnover_by_date_range
    )
    from backend.prompt import get_formatted_financial_assistant_prompt
    from backend.config import OLLAMA_API_URL, OLLAMA_MODEL
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Make sure your main.py, query_tools.py, prompt.py, and config.py are accessible.")
    exit(1)

def format_sql_query(query: str) -> str:
    """Format SQL query with proper indentation and line breaks."""
    import re

    # Common SQL keywords to add line breaks before
    keywords = ['SELECT', 'FROM', 'WHERE', 'ORDER BY', 'GROUP BY',
                'HAVING', 'JOIN', 'LEFT JOIN', 'RIGHT JOIN', 'INNER JOIN',
                'LIMIT', 'OFFSET', 'UNION', 'CASE', 'WHEN', 'ELSE', 'END',
                'WITH', 'AND', 'OR']

    # Ensure consistent spacing and casing
    formatted = query.strip()

    # Add line breaks before keywords
    for keyword in keywords:
        # Only replace if the keyword is a separate word (not part of another identifier)
        pattern = r'(?i)(\s+)(' + keyword + r')(\s+|$)'
        replacement = r'\n\2\3'
        formatted = re.sub(pattern, replacement, formatted)

    # Add indentation to lines after the first
    lines = formatted.split('\n')
    for i in range(1, len(lines)):
        lines[i] = '  ' + lines[i].strip()

    return '\n'.join(lines)

def extract_query(result: str) -> str:
    """Extract SQL query from the tool result."""
    if "Executing query: " in result:
        # Extract the query part
        query = result.split("Executing query: ", 1)[1].strip()
        # Format the query for better readability
        return format_sql_query(query)
    return result

def get_tool_descriptions() -> str:
    """Generate tool descriptions for the system prompt."""
    tools = [
        get_highest_lowest_close_prices_by_date_stock_number,
        get_technical_type_by_date_stock,
        get_stock_premium,
        get_top_stocks_by_date_and_weight,
        get_stock_holding_by_date_industry_ratio,
        worst_perform_stock_by_date,
        find_lowest_pb_hk_stock,
        find_weight_by_date,
        get_price_range_change,
        search_time_range_stock_report,
        get_stock_turnover_by_date_range
    ]

    tool_descriptions = []

    for tool in tools:
        # Get tool name
        name = tool.name if hasattr(tool, 'name') else (tool.__name__ if hasattr(tool, '__name__') else "Unknown")

        # Get description
        description = tool.description if hasattr(tool, 'description') else "No description"

        # Get parameter schema
        parameters = {}

        # Try to get args_schema
        schema = getattr(tool, 'args_schema', None)

        # Extract parameters from schema
        if schema:
            # Get fields - handle both Pydantic v1 and v2
            fields = {}
            if hasattr(schema, 'model_fields'):  # Pydantic V2
                fields = schema.model_fields
            elif hasattr(schema, '__fields__'):  # Pydantic V1 (deprecated)
                fields = schema.__fields__

            for field_name, field in fields.items():
                # Try to get field description
                description = "No description"
                if hasattr(field, 'description'):
                    description = field.description
                elif hasattr(field, 'field_info') and hasattr(field.field_info, 'description'):
                    description = field.field_info.description

                parameters[field_name] = description

        # Format tool description
        tool_desc = f"Tool: {name}\nDescription: {description}\nParameters: {json.dumps(parameters, ensure_ascii=False, indent=2)}"
        tool_descriptions.append(tool_desc)

    return "\n\n".join(tool_descriptions)

def get_tool_description_by_name(tool_name: str) -> str:
    """Get the description of a specific tool by name."""
    tools = [
        get_highest_lowest_close_prices_by_date_stock_number,
        get_technical_type_by_date_stock,
        get_stock_premium,
        get_top_stocks_by_date_and_weight,
        get_stock_holding_by_date_industry_ratio,
        worst_perform_stock_by_date,
        find_lowest_pb_hk_stock,
        find_weight_by_date,
        get_price_range_change,
        search_time_range_stock_report,
        get_stock_turnover_by_date_range
    ]

    for tool in tools:
        if hasattr(tool, 'name') and tool.name == tool_name:
            return tool.description if hasattr(tool, 'description') else "No description"

    return "Unknown tool"

def generate_financial_sql(unified_request: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process a UnifiedGenSQLRequest and generate a UnifiedGenSQLResponse.

    Args:
        unified_request: A dictionary with:
            - query_id: str, a unique identifier for the query
            - query: str, the natural language query

    Returns:
        A dictionary with:
            - query_id: str, the same ID from the request
            - query: str, the original query
            - sql: str, the formatted SQL query
            - tool_name: str, the name of the selected tool
            - parameters: Dict, the parameters used in the tool
            - tool_description: str, description of the tool
    """
    # Extract data from request
    query_id = unified_request.get("query_id", str(uuid.uuid4()))
    query = unified_request.get("query", "")

    # Prepare default response
    unified_response = {
        "query_id": query_id,
        "query": query,
        "sql": "",
        "tool_name": "",
        "parameters": {},
        "tool_description": ""
    }

    # If query is empty, return default response
    if not query:
        unified_response["tool_description"] = "Error: Empty query"
        return unified_response

    try:
        # Get tool descriptions for the system prompt
        tool_descriptions = get_tool_descriptions()

        # Create system prompt
        system_prompt = get_formatted_financial_assistant_prompt(tool_descriptions)

        # Create messages for LLM
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]

        # Get LLM response
        response_content, tool_name, parameters = chat_with_llm(query, messages)

        if tool_name:
            # Execute the selected tool
            result = execute_tool(tool_name, parameters)

            # Extract SQL query if present
            sql_query = ""
            if isinstance(result, str) and "Executing query: " in result:
                sql_query = extract_query(result)
            elif isinstance(result, dict) and "query" in result:
                sql_query = format_sql_query(result["query"])

            # Get tool description
            tool_description = get_tool_description_by_name(tool_name)

            # Update response
            unified_response.update({
                "sql": sql_query,
                "tool_name": tool_name,
                "parameters": parameters,
                "tool_description": tool_description
            })

        return unified_response

    except Exception as e:
        # Handle any errors
        unified_response["tool_description"] = f"Error: {str(e)}"
        traceback.print_exc()
        return unified_response

def run_batch_test():
    """Run batch testing with a list of predefined questions."""
    print("\n===== Financial Router Batch Test =====")
    print(f"Using model: {OLLAMA_MODEL}")
    print(f"API URL: {OLLAMA_API_URL}")

    # Sample financial questions for testing
    test_questions = [
        "What were the highest, lowest, and closing prices for Apple stock on 2023-04-15?",
        "What was the technical pattern for Google stock on 2023-05-10?",
        "What is the P/E ratio forecast for 中國銀行 on 2023-06-01?",
        "What are the top 5 stocks by earnings yield in the HSI on 2023-03-15?",
        "How many banking stocks are in the HSI index as of 2023-02-20 and what's their total weight?",
        "Which stock performed worst in the HSI index on 2023-01-15?",
        "What are the 3 Hong Kong banking stocks with the lowest P/B ratio on 2023-04-01?",
        "Which sectors have more than 10% weight in the HSI index on 2023-05-15?",
        "How much did the HSI index change between 2023-01-01 and 2023-03-31?",
        "What was the return for Tencent stock between 2023-01-01 and 2023-06-30?"
    ]

    # Adding Chinese questions (optional)
    chinese_test_questions = [
        "在2023-04-15這日，蘋果公司的股票最高價、最低價和收盤價是多少？",
        "在2023-05-10這日，谷歌股票的技術形態如何？",
        "在2023-06-01這日，中國銀行的預測市盈率是多少？",
        "在2023-03-15這日，恒生指數中收益率最高的5隻股票是哪些？",
        "截至2023-02-20，恒生指數中有多少銀行股，它們的總權重是多少？"
    ]

    # Use either English questions or both sets
    questions_to_test = test_questions  # or test_questions + chinese_test_questions

    try:
        print(f"\nProcessing {len(questions_to_test)} test questions...")

        # Process each question
        results = []
        for i, question in enumerate(questions_to_test):
            print(f"\nProcessing question {i+1}/{len(questions_to_test)}:")

            # Create request
            request = {
                "query_id": str(uuid.uuid4()),
                "query": question
            }

            # Display request
            print("\nINPUT:")
            print("UnifiedGenSQLRequest:")
            print(json.dumps(request, indent=4, ensure_ascii=False))

            # Generate response
            response = generate_financial_sql(request)

            # Display response
            print("\nOUTPUT:")
            print("UnifiedGenSQLResponse:")
            print(json.dumps(response, indent=4, ensure_ascii=False))
            print("="*50)

            results.append(response)

        # Save results to file
        with open("financial_queries_results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print("\nResults saved to financial_queries_results.json")

        # Display statistics
        tools_used = {}
        for result in results:
            tool = result["tool_name"] or "None"
            tools_used[tool] = tools_used.get(tool, 0) + 1

        print("\nTool usage statistics:")
        for tool, count in sorted(tools_used.items(), key=lambda x: x[1], reverse=True):
            print(f"  {tool}: {count}")

    except Exception as e:
        print(f"Error in batch test: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    run_batch_test()