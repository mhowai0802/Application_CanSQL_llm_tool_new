"""
Main LLM + Tools application for financial queries.
Uses direct LLM control flow rather than an agent.
"""
import streamlit as st
import re
import json
import requests
from typing import Dict, List, Any, Optional, Tuple
from query_tools import get_financial_query_tools, PARAMETER_DESCRIPTIONS
from config import OLLAMA_API_URL, OLLAMA_MODEL
from prompt import get_formatted_financial_assistant_prompt
from langchain_community.llms import Ollama

ollama_llm = Ollama(base_url=OLLAMA_API_URL, model=OLLAMA_MODEL)


# Function to send requests to Ollama API directly
def get_ollama_response(messages, model=OLLAMA_MODEL):
    """Send a request to Ollama API and get the response using LangChain's Ollama."""
    formatted_prompt = ""
    for msg in messages:
        role = msg['role']
        content = msg['content']
        if role == "system":
            formatted_prompt += f"<s>[SYSTEM] {content}</s>\n\n"
        elif role == "user":
            formatted_prompt += f"[USER] {content}\n\n"
        elif role == "assistant":
            formatted_prompt += f"[ASSISTANT] {content}\n\n"

    formatted_prompt += "[ASSISTANT] "

    try:
        response = ollama_llm(formatted_prompt)

        if not response or not response.strip():
            raise Exception("Empty response from LLM")

        return {"message": {"content": response}}
    except Exception as e:
        raise Exception(f"API error: {str(e)}")

# Get tools and generate tool descriptions for system prompt
@st.cache_data
def get_tool_descriptions() -> str:
    """Generate description of all available tools and their parameters."""
    tools = get_financial_query_tools()
    descriptions = []

    for tool in tools:
        tool_name = tool.name
        description = tool.description
        # Get parameters from the _run method signature
        import inspect
        signature = inspect.signature(tool._run)
        params = [p for p in signature.parameters]
        param_info = {}
        for param in params:
            param_info[param] = PARAMETER_DESCRIPTIONS.get(param, "No description available")

        descriptions.append(f"Tool: {tool_name}\nDescription: {description}\nParameters: {json.dumps(param_info, indent=2)}")

    return "\n\n".join(descriptions)

# Extract tool selection and parameters from LLM response
def parse_llm_response(response: str) -> Tuple[Optional[str], Dict[str, Any]]:
    """Parse LLM response to extract tool name and parameters."""
    try:
        # Try to find and parse JSON in the response
        json_match = re.search(r'({.*})', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
            data = json.loads(json_str)
            tool_name = data.get("tool")
            parameters = data.get("parameters", {})
            return tool_name, parameters
        else:
            return None, {}
    except Exception as e:
        st.error(f"Error parsing LLM response: {str(e)}")
        return None, {}

# Execute the selected tool with parameters
def execute_tool(tool_name: str, parameters: Dict[str, Any]) -> str:
    """Execute the selected tool with the provided parameters."""
    tools = get_financial_query_tools()
    tools_dict = {tool.name: tool for tool in tools}

    if tool_name not in tools_dict:
        return f"Error: Tool '{tool_name}' not found. Available tools: {', '.join(tools_dict.keys())}"

    try:
        tool = tools_dict[tool_name]
        result = tool._run(**parameters)
        return result
    except Exception as e:
        return f"Error executing tool: {str(e)}"

# Chat with the LLM and execute tools
def chat_with_llm(prompt: str, messages: List[Dict[str, str]]) -> Tuple[str, Optional[str], Dict[str, Any]]:
    """
    Chat with the LLM to identify the appropriate tool and parameters.
    Returns the LLM response, selected tool name, and parameters.
    """
    try:
        response = get_ollama_response(messages)

        response_content = response["message"]["content"]
        tool_name, parameters = parse_llm_response(response_content)

        return response_content, tool_name, parameters
    except Exception as e:
        return f"Error: {str(e)}", None, {}

# Main UI
def main():
    st.set_page_config(page_title="Financial Data Assistant", layout="wide")

    st.title("Financial Data Assistant")

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "system_prompt" not in st.session_state:
        tool_descriptions = get_tool_descriptions()
        st.session_state.system_prompt = get_formatted_financial_assistant_prompt(tool_descriptions)

    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")

        st.subheader("Model Information")
        st.info(f"Using model: {OLLAMA_MODEL}")
        st.info(f"API URL: {OLLAMA_API_URL}")

        st.subheader("System Prompt")
        system_prompt = st.text_area("System Prompt", st.session_state.system_prompt, height=300)
        if system_prompt != st.session_state.system_prompt:
            st.session_state.system_prompt = system_prompt

        st.subheader("Available Tools")
        tools = get_financial_query_tools()
        tool_info = []
        for tool in tools:
            tool_info.append(f"**{tool.name}**: {tool.description}")
        st.markdown("\n\n".join(tool_info))

    # Display chat history
    for message in st.session_state.messages:
        role = message["role"]
        content = message["content"]

        with st.chat_message(role):
            st.write(content)

            # Show tool execution results if available
            if "tool_name" in message and "parameters" in message:
                tool_name = message.get("tool_name")
                parameters = message.get("parameters", {})

                if tool_name:
                    with st.expander("Tool Execution Details"):
                        st.markdown(f"**Selected Tool:** {tool_name}")
                        st.markdown("**Parameters:**")
                        st.json(parameters)

                        if "tool_result" in message:
                            st.markdown("**Execution Result:**")
                            st.code(message["tool_result"])

    # Chat input
    prompt = st.chat_input("Ask a financial data question...")

    if prompt:
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message
        with st.chat_message("user"):
            st.write(prompt)

        # Prepare messages for LLM
        messages = [
            {"role": "system", "content": st.session_state.system_prompt}
        ]

        # Add chat history (limited to last 10 exchanges)
        for msg in st.session_state.messages[-10:]:
            if msg["role"] in ["user", "assistant"]:
                messages.append({"role": msg["role"], "content": msg["content"]})

        # Get LLM response and tool selection
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response_content, tool_name, parameters = chat_with_llm(prompt, messages)
                st.write(response_content)

                # If a tool was selected, execute it
                tool_result = None
                if tool_name:
                    with st.status(f"Executing tool: {tool_name}"):
                        tool_result = execute_tool(tool_name, parameters)

                    st.write("Tool execution result:")
                    st.code(tool_result)

        # Add assistant message to chat history
        assistant_message = {
            "role": "assistant",
            "content": response_content,
            "tool_name": tool_name,
            "parameters": parameters
        }

        if tool_result:
            assistant_message["tool_result"] = tool_result

        st.session_state.messages.append(assistant_message)

if __name__ == "__main__":
    main()