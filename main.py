import streamlit as st
import re
import json
import sys
from typing import Dict, List, Any, Optional, Tuple
import inspect
import importlib
from langchain_community.llms import Ollama
from config import OLLAMA_API_URL, OLLAMA_MODEL
from prompt import get_formatted_financial_assistant_prompt

# Import query_tools module
print("Importing query_tools module...")
try:
    import query_tools

    print(f"Successfully imported query_tools")

    # Find all LangChain tools
    tools_list = []
    for name in dir(query_tools):
        attr = getattr(query_tools, name)
        # Check if it's a LangChain tool (they are instances of classes like StructuredTool)
        if hasattr(attr, 'name') and hasattr(attr, 'description') and callable(getattr(attr, 'invoke', None)):
            tools_list.append(attr)
            print(f"Found LangChain tool: {attr.name}")

    print(f"Found {len(tools_list)} LangChain tools")
except Exception as e:
    print(f"Error importing query_tools: {str(e)}")
    st.error(f"Failed to import tools: {str(e)}")
    tools_list = []

# Initialize Ollama LLM
print(f"Initializing Ollama LLM with URL: {OLLAMA_API_URL} and model: {OLLAMA_MODEL}")
ollama_llm = Ollama(base_url=OLLAMA_API_URL, model=OLLAMA_MODEL)
print("Ollama LLM initialized")


# Function to decode unicode escape sequences in parameters
def decode_parameters(parameters: Dict[str, Any]) -> Dict[str, Any]:
    """Decode any Unicode escape sequences in parameter values."""
    print(f"Decoding parameters: {parameters}")
    decoded_params = {}
    for key, value in parameters.items():
        if isinstance(value, str):
            # No need to do additional decoding as Python handles it
            decoded_params[key] = value
        else:
            decoded_params[key] = value
    print(f"Decoded parameters: {decoded_params}")
    return decoded_params


# Format tool name for display
def format_tool_name(tool_name: str) -> str:
    """Format the tool name to be more readable."""
    print(f"Formatting tool name: {tool_name}")
    if not tool_name:
        print("Empty tool name, returning 'Unknown Tool'")
        return "Unknown Tool"

    # Remove _tool suffix if present
    name = tool_name.replace('_tool', '')

    # Split by underscores and capitalize each word
    words = name.split('_')

    # Capitalize words
    formatted_words = [word.capitalize() for word in words]

    result = " ".join(formatted_words)
    print(f"Formatted tool name: {result}")
    return result


# Function to send requests to Ollama API directly
def get_ollama_response(messages, model=OLLAMA_MODEL):
    """Send a request to Ollama API and get the response using LangChain's Ollama."""
    print(f"Sending request to Ollama API with {len(messages)} messages")
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
    print(f"Formatted prompt length: {len(formatted_prompt)} characters")

    try:
        print("Calling Ollama LLM...")
        response = ollama_llm(formatted_prompt)
        print(f"Received response length: {len(response) if response else 0} characters")

        if not response or not response.strip():
            print("Warning: Empty response from LLM")
            raise Exception("Empty response from LLM")

        return {"message": {"content": response}}
    except Exception as e:
        print(f"Error in get_ollama_response: {str(e)}")
        raise Exception(f"API error: {str(e)}")


# Get available tools - without using cache
def get_available_tools():
    """Get all available LangChain tools with serializable info."""
    serializable_tools = []

    for tool in tools_list:
        try:
            # Extract only the serializable information
            tool_info = {
                "name": tool.name,
                "description": tool.description
            }

            # Get schema info from the tool if available
            schema = {}
            if hasattr(tool, 'args_schema'):
                schema_class = tool.args_schema
                if hasattr(schema_class, 'model_fields'):
                    for field_name, field in schema_class.model_fields.items():
                        schema[field_name] = {
                            "description": field.description or f"Parameter {field_name}",
                            "type": str(field.annotation)
                        }
                elif hasattr(schema_class, '__fields__'):  # For older Pydantic
                    for field_name, field in schema_class.__fields__.items():
                        schema[field_name] = {
                            "description": field.field_info.description or f"Parameter {field_name}",
                            "type": str(field.type_)
                        }

            tool_info["schema"] = schema
            serializable_tools.append(tool_info)

        except Exception as e:
            print(f"Error processing tool {tool}: {str(e)}")

    return serializable_tools


# Get tool descriptions for system prompt - without using cache
def get_tool_descriptions() -> str:
    """Generate description of all available tools and their parameters."""
    tools = get_available_tools()
    descriptions = []

    for tool in tools:
        tool_name = tool["name"]
        description = tool["description"]
        schema = tool["schema"]

        descriptions.append(
            f"Tool: {tool_name}\nDescription: {description}\nParameters: {json.dumps(schema, indent=2)}")

    result = "\n\n".join(descriptions)
    return result


# Get tool by name
def get_tool_by_name(tool_name):
    """Find a tool by its name from the tools_list."""
    for tool in tools_list:
        if tool.name == tool_name:
            return tool
    return None


# Extract tool selection and parameters from LLM response
def parse_llm_response(response: str) -> Tuple[Optional[str], Dict[str, Any]]:
    """Parse LLM response to extract tool name and parameters."""
    print(f"Parsing LLM response with length: {len(response)} characters")
    print(f"Response preview: {response[:100]}...")

    try:
        # Try to find and parse JSON in the response
        json_match = re.search(r'({.*})', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
            print(f"Found JSON: {json_str}")
            data = json.loads(json_str)
            tool_name = data.get("tool")
            parameters = data.get("parameters", {})
            print(f"Extracted tool: {tool_name}")
            print(f"Extracted parameters: {parameters}")

            # Decode any Unicode escape sequences
            parameters = decode_parameters(parameters)

            return tool_name, parameters
        else:
            print("No JSON found in response")
            return None, {}
    except Exception as e:
        print(f"Error parsing LLM response: {str(e)}")
        st.error(f"Error parsing LLM response: {str(e)}")
        return None, {}


# Execute the selected tool with parameters
def execute_tool(tool_name: str, parameters: Dict[str, Any]) -> str:
    """Execute the selected tool with the provided parameters."""
    print(f"Executing tool: {tool_name} with parameters: {parameters}")

    tool = get_tool_by_name(tool_name)
    if not tool:
        error_msg = f"Error: Tool '{tool_name}' not found. Available tools: {', '.join([t.name for t in tools_list])}"
        print(error_msg)
        return error_msg

    try:
        print(f"Invoking LangChain tool with parameters: {parameters}")
        result = tool.invoke(parameters)
        print(f"Tool execution successful, result: {result}")
        return result
    except Exception as e:
        error_msg = f"Error executing tool: {str(e)}"
        print(f"Error executing tool: {str(e)}")
        print(f"Exception type: {type(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return error_msg


# Chat with the LLM and execute tools
def chat_with_llm(prompt: str, messages: List[Dict[str, str]]) -> Tuple[str, Optional[str], Dict[str, Any]]:
    """
    Chat with the LLM to identify the appropriate tool and parameters.
    Returns the LLM response, selected tool name, and parameters.
    """
    print(f"Chatting with LLM using prompt: {prompt[:50]}...")
    try:
        response = get_ollama_response(messages)
        response_content = response["message"]["content"]
        print(f"Got response from LLM, length: {len(response_content)} characters")
        tool_name, parameters = parse_llm_response(response_content)
        return response_content, tool_name, parameters
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        print(f"Error in chat_with_llm: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return error_msg, None, {}


# Display tool information using Streamlit components
def display_tool_info(tool_name: str, parameters: Dict[str, Any]):
    """Display tool and parameter information using Streamlit components."""
    print(f"Displaying tool info for: {tool_name}")
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
        print(f"Displaying parameters: {parameters}")

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


# Main UI
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
        print("Generating system prompt")
        try:
            tool_descriptions = get_tool_descriptions()
            print("Successfully generated tool descriptions")
            st.session_state.system_prompt = get_formatted_financial_assistant_prompt(tool_descriptions)
            print("Successfully generated formatted system prompt")
        except Exception as e:
            print(f"Error generating system prompt: {str(e)}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            st.error(f"Error initializing system: {str(e)}")
            st.session_state.system_prompt = "Error loading tools. Please check the logs."

    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")

        st.subheader("Model Information")
        st.info(f"Using model: {OLLAMA_MODEL}")
        st.info(f"API URL: {OLLAMA_API_URL}")

        st.info(f"Found {len(tools_list)} available tools")

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

    # Display chat history
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

        print(f"Prepared {len(messages)} messages for LLM")

        # Get LLM response and tool selection
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                print("Calling LLM for response")
                response_content, tool_name, parameters = chat_with_llm(prompt, messages)

                # Remove JSON from display
                clean_content = re.sub(r'({.*})', '', response_content).strip()
                if clean_content:
                    print(f"Displaying clean content (length: {len(clean_content)})")
                    st.write(clean_content)

                # If a tool was selected, display its info and execute it
                if tool_name:
                    print(f"Tool '{tool_name}' selected, displaying info and executing")
                    # Display tool info using Streamlit components
                    display_tool_info(tool_name, parameters)

                    # Execute the tool
                    with st.status("Running query..."):
                        print(f"Executing tool: {tool_name}")
                        tool_result = execute_tool(tool_name, parameters)

                    # Display results
                    st.markdown("**Results:**")
                    st.code(tool_result)
                    tool_result_to_save = tool_result
                else:
                    print("No tool selected")
                    tool_result_to_save = None

        # Add assistant message to chat history
        print("Adding assistant message to chat history")
        assistant_message = {
            "role": "assistant",
            "content": response_content,
            "tool_name": tool_name,
            "parameters": parameters
        }

        if tool_result_to_save:
            assistant_message["tool_result"] = tool_result_to_save

        st.session_state.messages.append(assistant_message)


if __name__ == "__main__":
    print("=== Starting Financial Data Assistant Application ===")
    main()