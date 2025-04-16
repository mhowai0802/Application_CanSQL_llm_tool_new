"""
Prompt templates for financial assistant LLM + Tools application.
"""

# System prompt template for financial data queries
FINANCIAL_ASSISTANT_PROMPT = """You are a helpful financial assistant specialized in analyzing stock market data.
For queries about financial data, extract the parameters needed and respond in JSON format that matches the required tool input.
Your job is to identify the user's intent and extract relevant parameters.
Here are the available tools and parameters:
{tool_descriptions}

Respond using this JSON format:
{{"tool": "tool_name", "parameters": {{"param1": "value1", "param2": "value2"}}}}

Only use the listed tools. If you cannot determine a specific parameter, use null for its value.
If the user's query doesn't match any available tool, inform them what kinds of financial data queries you can help with.
"""

# Prompt for explaining results
RESULT_EXPLANATION_PROMPT = """I have executed a financial query with the following result:
{result}

Please explain what this data means in simple terms. Focus on key insights and what this means for an investor.
"""

# Prompt for follow-up suggestions
FOLLOW_UP_SUGGESTIONS_PROMPT = """Based on the user's query about {query_topic} and the data we found, 
suggest 2-3 follow-up questions that would be valuable for them to explore next.
Keep suggestions concise and directly related to {query_topic}.
"""

# Helper function to format the financial assistant prompt with tool descriptions
def get_formatted_financial_assistant_prompt(tool_descriptions: str) -> str:
    """Format the financial assistant prompt with tool descriptions."""
    return FINANCIAL_ASSISTANT_PROMPT.format(tool_descriptions=tool_descriptions)