import streamlit as st
import requests
import re
import json
from typing import Dict, List, Any, Optional, Tuple
import sys

# API endpoint
API_URL = "http://localhost:5000/api"


def format_tool_name(tool_name: str) -> str:
    """Format the tool name to be more readable."""
    if not tool_name:
        return "Unknown Tool"

    # Remove _tool suffix if present
    name = tool_name.replace('_tool', '')

    # Split by underscores and capitalize each word
    words = name.split('_')

    # Capitalize words
    formatted_words = [word.capitalize() for word in words]

    return " ".join(formatted_words)


def display_tool_info(tool_name: str, parameters: Dict[str, Any]):
    """Display tool and parameter information using Streamlit components."""
    formatted_tool_name = format_tool_name(tool_name)

    # Tool name in a colored box
    st.markdown(f"""
    <div style="background-color:#e6f3ff; padding:10px; border-radius:5px; margin-bottom:10px">
        <h7 style="margin:0; color:#0066cc;">🔧 {formatted_tool_name}</h7>
    </div>
    """, unsafe_allow_html=True)

    # Parameters table
    if parameters:
        st.write("Parameters:")

        # Create two columns
        col1, col2 = st.columns(2)

        # Display parameters in the columns
        for i, (key, value) in enumerate(parameters.items()):
            # Format the key for display
            display_key = " ".join(word.capitalize() for word in key.split('_'))

            # Alternate between columns
            if i % 2 == 0:
                with col1:
                    st.info(f"**{display_key}**: {value}")
            else:
                with col2:
                    st.info(f"**{display_key}**: {value}")


def get_model_info():
    """Get model information from the API."""
    try:
        response = requests.get(f"{API_URL}/model_info")
        return response.json()
    except Exception as e:
        st.error(f"Error getting model info: {str(e)}")
        return {"model": "Unknown", "api_url": "Unknown"}


def get_available_tools():
    """Get available tools from the API."""
    try:
        response = requests.get(f"{API_URL}/tools")
        data = response.json()
        return data.get("tools", [])
    except Exception as e:
        st.error(f"Error getting tools: {str(e)}")
        return []


def get_system_prompt():
    """Get system prompt from the API."""
    try:
        response = requests.get(f"{API_URL}/system_prompt")
        data = response.json()
        return data.get("system_prompt", "")
    except Exception as e:
        st.error(f"Error getting system prompt: {str(e)}")
        return "Error loading system prompt."


def chat_with_llm(prompt, system_prompt):
    """Send a chat request to the API without chat history."""
    try:
        # Only send the current prompt and system prompt, no chat history
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]

        response = requests.post(
            f"{API_URL}/chat",
            json={"prompt": prompt, "messages": messages}
        )
        return response.json()
    except Exception as e:
        st.error(f"Error chatting with LLM: {str(e)}")
        return {"error": str(e)}


def main():
    print("Starting Financial Data Assistant app")
    print(f"Python version: {sys.version}")

    st.set_page_config(page_title="Financial Data Assistant", layout="wide")

    st.title("Financial Data Assistant")

    # Initialize session state
    if "messages" not in st.session_state:
        print("Initializing messages in session state")
        st.session_state.messages = []

    if "system_prompt" not in st.session_state:
        print("Getting system prompt")
        try:
            st.session_state.system_prompt = get_system_prompt()
            print("Successfully retrieved system prompt")
        except Exception as e:
            print(f"Error getting system prompt: {str(e)}")
            st.error(f"Error initializing system: {str(e)}")
            st.session_state.system_prompt = "Error loading tools. Please check the logs."

    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")

        st.subheader("Model Information")
        model_info = get_model_info()
        st.info(f"Using model: {model_info.get('model')}")
        st.info(f"API URL: {model_info.get('api_url')}")

        # Add a notice about stateless operation
        st.warning(
            "⚠️ This app operates in stateless mode. Each query is independent and doesn't consider previous conversation context.")

        st.subheader("System Prompt")
        system_prompt = st.text_area("System Prompt", st.session_state.system_prompt, height=300)
        if system_prompt != st.session_state.system_prompt:
            print("System prompt updated by user")
            st.session_state.system_prompt = system_prompt

        st.subheader("Available Tools")
        try:
            print("Getting available tools for sidebar")
            tools = get_available_tools()
            if len(tools) == 0:
                st.warning("No tools available. Please check console for errors.")

            tool_info = []
            for tool in tools:
                tool_info.append(f"**{tool['name']}**: {tool['description']}")
            st.markdown("\n\n".join(tool_info))
        except Exception as e:
            print(f"Error displaying tools in sidebar: {str(e)}")
            st.error(f"Error loading tools: {str(e)}")

    # Display chat history (stored only in frontend)
    print(f"Displaying {len(st.session_state.messages)} messages in chat history")
    for message in st.session_state.messages:
        role = message["role"]
        content = message["content"]

        with st.chat_message(role):
            # Display the raw LLM response without the JSON part
            if role == "assistant":
                # Remove JSON from display
                clean_content = re.sub(r'({.*})', '', content).strip()
                if clean_content:
                    st.write(clean_content)

                # Display tool info using Streamlit components if available
                if "tool_name" in message and "parameters" in message:
                    tool_name = message.get("tool_name")
                    parameters = message.get("parameters", {})
                    if tool_name:
                        print(f"Displaying tool info in chat history: {tool_name}")
                        display_tool_info(tool_name, parameters)
            else:
                st.write(content)

            # Show query results if available
            if "tool_result" in message:
                st.markdown("**Results:**")
                st.code(message["tool_result"])

    # Chat input
    prompt = st.chat_input("Ask a financial data question...")

    if prompt:
        print(f"Received user prompt: {prompt}")
        # Add user message to chat history (frontend only)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message
        with st.chat_message("user"):
            st.write(prompt)

        # Get LLM response and tool selection (stateless - no chat history)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                print("Calling LLM for response (stateless)")

                # Call backend with only current prompt and system prompt
                response_data = chat_with_llm(prompt, st.session_state.system_prompt)

                if "error" in response_data:
                    st.error(response_data["error"])
                    return

                response_content = response_data.get("response", "")
                tool_name = response_data.get("tool_name")
                parameters = response_data.get("parameters", {})
                tool_result = response_data.get("tool_result")

                # Remove JSON from display
                clean_content = re.sub(r'({.*})', '', response_content).strip()
                if clean_content:
                    print(f"Displaying clean content (length: {len(clean_content)})")
                    st.write(clean_content)

                # If a tool was selected, display its info
                if tool_name:
                    print(f"Tool '{tool_name}' selected, displaying info")
                    # Display tool info using Streamlit components
                    display_tool_info(tool_name, parameters)

                    # Display results
                    if tool_result:
                        st.markdown("**Results:**")
                        st.code(tool_result)
                else:
                    print("No tool selected")

        # Add assistant message to chat history (frontend only)
        print("Adding assistant message to frontend chat history")
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
    print("=== Starting Financial Data Assistant Application ===")
    main()