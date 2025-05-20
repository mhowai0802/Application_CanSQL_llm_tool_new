from flask import Flask, request, jsonify
import re
import json
import traceback
from typing import Dict, List, Any, Optional, Tuple
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
    tools_list = []

# Initialize Ollama LLM
print(f"Initializing Ollama LLM with URL: {OLLAMA_API_URL} and model: {OLLAMA_MODEL}")
ollama_llm = Ollama(base_url=OLLAMA_API_URL, model=OLLAMA_MODEL)
print("Ollama LLM initialized")

app = Flask(__name__)


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


# Get tool by name
def get_tool_by_name(tool_name):
    """Find a tool by its name from the tools_list."""
    print(tool_name, tools_list)
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
        print(f"Traceback: {traceback.format_exc()}")
        return error_msg


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
        print(f"Traceback: {traceback.format_exc()}")
        return error_msg, None, {}


# API endpoints
@app.route('/api/tools', methods=['GET'])
def get_tools():
    """API endpoint to get all available tools."""
    try:
        tools = get_available_tools()
        return jsonify({"tools": tools})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/system_prompt', methods=['GET'])
def get_system_prompt():
    """API endpoint to get the system prompt."""
    try:
        tool_descriptions = get_tool_descriptions()
        system_prompt = get_formatted_financial_assistant_prompt(tool_descriptions)
        return jsonify({"system_prompt": system_prompt})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/chat', methods=['POST'])
def chat():
    """API endpoint to chat with the LLM."""
    try:
        data = request.json
        prompt = data.get('prompt')
        messages = data.get('messages', [])

        if not prompt:
            return jsonify({"error": "No prompt provided"}), 400

        response_content, tool_name, parameters = chat_with_llm(prompt, messages)

        result = {
            "response": response_content,
            "tool_name": tool_name,
            "parameters": parameters
        }

        # If a tool was selected, execute it
        if tool_name:
            tool_result = execute_tool(tool_name, parameters)
            result["tool_result"] = tool_result

        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/execute_tool', methods=['POST'])
def execute_tool_api():
    """API endpoint to execute a tool."""
    try:
        data = request.json
        tool_name = data.get('tool_name')
        parameters = data.get('parameters', {})

        if not tool_name:
            return jsonify({"error": "No tool name provided"}), 400

        result = execute_tool(tool_name, parameters)
        return jsonify({"result": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/model_info', methods=['GET'])
def get_model_info():
    """API endpoint to get model information."""
    return jsonify({
        "model": OLLAMA_MODEL,
        "api_url": OLLAMA_API_URL
    })


if __name__ == '__main__':
    print("=== Starting Financial Data Assistant API ===")
    app.run(debug=True, port=5000)