import requests
import boto3
import json
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field

# VirusTotal API functions (unchanged from your original code)
def get_ip_report_from_virustotal(ip_address):
    headers = {
        "x-apikey": virustotal_api_key,
        "Content-Type": "application/json"
    }
    response = requests.get(f"{virustotal_api_url}/ip_addresses/{ip_address}", headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        return None

def get_file_report_from_virustotal(hash_file):
    headers = {
        "x-apikey": virustotal_api_key,
        "Content-Type": "application/json"
    }
    response = requests.get(f"{virustotal_api_url}/files/{hash_file}", headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        return None

def get_url_report_from_virustotal(url):
    payload = {"url": url}
    headers1 = {
        "accept": "application/json",
        "x-apikey": virustotal_api_key,
        "Content-Type": "application/x-www-form-urlencoded"
    }
    scan_url = requests.post(f"{virustotal_api_url}/urls", data=payload, headers=headers1)
    print(scan_url.text)
    return scan_url.json() if scan_url.status_code == 200 else None

def get_subdomains_from_virustotal(domain):
    headers = {
        "x-apikey": virustotal_api_key,
        "Content-Type": "application/json"
    }
    response = requests.get(f"{virustotal_api_url}/domains/{domain}/subdomains", headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        return None

def get_files_from_virustotal(domain):
    headers = {
        "x-apikey": virustotal_api_key,
        "Content-Type": "application/json"
    }
    response = requests.get(f"{virustotal_api_url}/domains/{domain}/communicating_files", headers=headers)
    
    if response.status_code == 200:
        report = response.json()
        data = report["data"]
        file_details_list = []
        for file in data:
            file_attributes = file["attributes"]
            file_details = {
                "file_hash": file.get("id"),
                "file_name": file_attributes.get("names", ["Unknown"]),
                "file_type": file_attributes.get("type_description", "Unknown"),
                "file_size": file_attributes.get("size", "Unknown"),
                "last_analysis_date": file_attributes.get("last_analysis_date", "Unknown"),
                "detection": file_attributes.get("last_analysis_stats", {})
            }
            file_details_list.append(file_details)
        
        return file_details_list
    else:
        return None

def get_dns_domain_from_virustotal(domain):
    headers = {
        "x-apikey": virustotal_api_key,
        "Content-Type": "application/json"
    }
    response = requests.get(f"{virustotal_api_url}/domains/{domain}/resolutions", headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        return None

# Tool definitions for Bedrock
def create_tool_definitions():
    """Create tool definitions in the format expected by Bedrock"""
    return [
        {
            "toolSpec": {
                "name": "get_ip_report_from_virustotal",
                "description": "Get the report for an IP address from VirusTotal",
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "ip_address": {
                                "type": "string",
                                "description": "The IP address to query"
                            }
                        },
                        "required": ["ip_address"]
                    }
                }
            }
        },
        {
            "toolSpec": {
                "name": "get_file_report_from_virustotal",
                "description": "Get the report for a file hash from VirusTotal",
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "hash_file": {
                                "type": "string",
                                "description": "The file hash to query"
                            }
                        },
                        "required": ["hash_file"]
                    }
                }
            }
        },
        {
            "toolSpec": {
                "name": "get_url_report_from_virustotal",
                "description": "Get the report for a URL from VirusTotal",
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "url": {
                                "type": "string",
                                "description": "The URL to query"
                            }
                        },
                        "required": ["url"]
                    }
                }
            }
        },
        {
            "toolSpec": {
                "name": "get_subdomains_from_virustotal",
                "description": "Get subdomains for a domain from VirusTotal",
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "domain": {
                                "type": "string",
                                "description": "The domain to query"
                            }
                        },
                        "required": ["domain"]
                    }
                }
            }
        },
        {
            "toolSpec": {
                "name": "get_files_from_virustotal",
                "description": "Get communicating files for a domain from VirusTotal",
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "domain": {
                                "type": "string",
                                "description": "The domain to query"
                            }
                        },
                        "required": ["domain"]
                    }
                }
            }
        },
        {
            "toolSpec": {
                "name": "get_dns_domain_from_virustotal",
                "description": "Get passive DNS resolutions for a domain from VirusTotal",
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "domain": {
                                "type": "string",
                                "description": "The domain to query"
                            }
                        },
                        "required": ["domain"]
                    }
                }
            }
        }
    ]

# Initialize Bedrock client using boto3
def initialize_bedrock_client():
    """Initialize the Bedrock client"""
    return boto3.client(
        'bedrock-runtime',
        region_name='us-east-1'  # Change to your preferred region
    )

def execute_tool(tool_name: str, parameters: dict) -> dict:
    """Execute the appropriate VirusTotal function based on tool name"""
    try:
        if tool_name == "get_ip_report_from_virustotal":
            return get_ip_report_from_virustotal(parameters.get("ip_address"))
        elif tool_name == "get_file_report_from_virustotal":
            return get_file_report_from_virustotal(parameters.get("hash_file"))
        elif tool_name == "get_url_report_from_virustotal":
            return get_url_report_from_virustotal(parameters.get("url"))
        elif tool_name == "get_subdomains_from_virustotal":
            return get_subdomains_from_virustotal(parameters.get("domain"))
        elif tool_name == "get_files_from_virustotal":
            return get_files_from_virustotal(parameters.get("domain"))
        elif tool_name == "get_dns_domain_from_virustotal":
            return get_dns_domain_from_virustotal(parameters.get("domain"))
        else:
            return {"error": f"Unknown tool: {tool_name}"}
    except Exception as e:
        return {"error": f"Error executing {tool_name}: {str(e)}"}

# Main function to handle conversations with tools using boto3 directly
def generate_bedrock_response(messages: List[Dict[str, str]]) -> str:
    """
    Generate response using Bedrock with tool calling capabilities
    
    Args:
        messages: List of message dictionaries with 'role' and 'content' keys
    
    Returns:
        str: The response from the model
    """
    
    try:
        # Initialize the Bedrock client
        bedrock = initialize_bedrock_client()
        
        # Prepare the conversation for Bedrock format
        bedrock_messages = []
        for msg in messages:
            if msg["role"] in ["user", "assistant"]:
                bedrock_messages.append({
                    "role": msg["role"],
                    "content": [{"text": msg["content"]}]
                })
        
        # Add system message if not present
        system_message = "You are a cybersecurity assistant that can analyze threats using VirusTotal. Use the available tools to help users with their security queries."
        
        # Tool configuration
        tool_config = {
            "tools": create_tool_definitions(),
            "toolChoice": {"auto": {}}
        }
        
        # First API call to get initial response
        response = bedrock.converse(
            modelId="anthropic.claude-3-5-sonnet-20241022-v2:0",
            messages=bedrock_messages,
            system=[{"text": system_message}],
            toolConfig=tool_config,
            inferenceConfig={
                "temperature": 0.1,
                "maxTokens": 4000
            }
        )
        
        # Check if the model wants to use tools
        if response['stopReason'] == 'tool_use':
            # Execute tool calls
            tool_results = []
            assistant_message = {
                "role": "assistant",
                "content": response['output']['message']['content']
            }
            bedrock_messages.append(assistant_message)
            
            # Process each tool use request
            for content_block in response['output']['message']['content']:
                if content_block.get('toolUse'):
                    tool_use = content_block['toolUse']
                    tool_name = tool_use['name']
                    tool_input = tool_use['input']
                    tool_id = tool_use['toolUseId']
                    
                    # Execute the tool
                    tool_result = execute_tool(tool_name, tool_input)
                    
                    # Add tool result to the conversation
                    tool_results.append({
                        "toolResult": {
                            "toolUseId": tool_id,
                            "content": [{"text": json.dumps(tool_result)}]
                        }
                    })
            
            # Add tool results as user message
            if tool_results:
                bedrock_messages.append({
                    "role": "user",
                    "content": tool_results
                })
                
                # Get final response after tool execution
                final_response = bedrock.converse(
                    modelId="anthropic.claude-3-5-sonnet-20241022-v2:0",
                    messages=bedrock_messages,
                    system=[{"text": system_message}],
                    toolConfig=tool_config,
                    inferenceConfig={
                        "temperature": 0.1,
                        "maxTokens": 4000
                    }
                )
                
                # Extract the text response
                final_text = ""
                for content in final_response['output']['message']['content']:
                    if content.get('text'):
                        final_text += content['text']
                
                return final_text
        
        # No tool use, return direct response
        response_text = ""
        for content in response['output']['message']['content']:
            if content.get('text'):
                response_text += content['text']
        
        return response_text
        
    except Exception as e:
        return f"Error generating response: {str(e)}"

# Backward compatibility functions for the API
get_ip_report_from_virustotal_tool = None
get_file_report_from_virustotal_tool = None
get_url_report_from_virustotal_tool = None
get_subdomains_from_virustotal_tool = None
get_files_from_virustotal_tool = None
get_dns_domain_from_virustotal_tool = None

# Example usage function (for testing the functions directly)
def example_usage():
    """Example of how to use the new Bedrock-based system"""
    
    # Example messages (similar to your OpenAI format)
    messages = [
        {
            "role": "user", 
            "content": "Can you check the reputation of IP address 8.8.8.8?"
        }
    ]
    
    response = generate_bedrock_response(messages)
    print("Response:", response)

# Configuration (you'll need to set these)
virustotal_api_key = "YOUR_VIRUSTOTAL_API_KEY"
virustotal_api_url = "https://www.virustotal.com/api/v3"

if __name__ == "__main__":
    example_usage()


# api.py - FastAPI endpoint file
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging
import json
from typing import List, Dict, Any

# Import the AWS Bedrock functions from your aws_functions.py file
from aws_functions import (
    generate_bedrock_response,
    initialize_bedrock_client,
    get_ip_report_from_virustotal,
    get_file_report_from_virustotal,
    get_url_report_from_virustotal,
    get_subdomains_from_virustotal,
    get_files_from_virustotal,
    get_dns_domain_from_virustotal,
    execute_tool,
    create_tool_definitions
)

# Setup logging
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(title="Cybersecurity Threat Analysis API", version="2.0.0")

class QueryRequest(BaseModel):
    user_id: str
    query: str

# Global conversation storage (in production, use a proper database)
last_conversation = []

# Simple conversation starters (same as your original)
simple_conversation_starters = {
    "hello": {"response": "Hello! How can I help you with cybersecurity analysis today?"},
    "hi": {"response": "Hi there! I'm here to help you analyze potential threats using VirusTotal."},
    "help": {"response": "I can help you analyze IPs, file hashes, URLs, and domains using VirusTotal. Just ask me to check something!"}
}

@app.post("/threat")
async def analyze_query(request: QueryRequest):
    """
    Analyze cybersecurity threats using AWS Bedrock and VirusTotal
    Simple implementation using the generate_bedrock_response function
    """
    global last_conversation
    
    print("client:", request)
    print("type(client):", type(request))
    
    lower_query = request.query.lower()
    logger.info(f"data: {request}")
    
    # Check for simple conversation starters
    if lower_query in simple_conversation_starters:
        predefined = simple_conversation_starters[lower_query]
        response = predefined["response"]
        return {"text": response, "type": "text"}
    
    else:
        # Update the conversation history
        last_conversation = last_conversation[-2:]  # Keep only the last conversation
        # Append the current query to the last conversation
        messages = last_conversation + [{"role": "user", "content": request.query}]
        last_conversation = last_conversation + [{"role": "user", "content": request.query}]
        
        try:
            # Generate response from AWS Bedrock (this handles tool calls internally)
            response = generate_bedrock_response(messages)
            final_answer = response
            
            # Store the response in conversation history
            last_conversation.append({"role": "assistant", "content": final_answer})
            last_conversation = last_conversation[-2:]  # Keep only last 2 messages
            
            # Store in database (implement your store_db functions)
            # store_db.store_query_response_threat(request.user_id, request.query, final_answer, "text")
            logger.info(f"chat response: {final_answer}")
            
            return {"text": final_answer, "type": "text"}
            
        except Exception as e:
            logger.error(f"Error in analyze_query: {str(e)}")
            final_answer = "Sorry, I couldn't retrieve the report."
            # store_db.store_query_response_threat(request.user_id, request.query, final_answer, "text")
            logger.info(f"chat response: {final_answer}")
            return {"text": final_answer, "type": "text"}

@app.post("/threat_detailed")
async def analyze_query_detailed(request: QueryRequest):
    """
    Alternative implementation using direct boto3 calls to Bedrock
    """
    global last_conversation
    
    reports = []
    lower_query = request.query.lower()
    logger.info(f"data: {request}")
    
    if lower_query in simple_conversation_starters:
        predefined = simple_conversation_starters[lower_query]
        response = predefined["response"]
        return {"text": response, "type": "text"}
    
    else:
        # Update conversation history
        last_conversation = last_conversation[-2:]
        messages = last_conversation + [{"role": "user", "content": request.query}]
        last_conversation = last_conversation + [{"role": "user", "content": request.query}]
        
        try:
            # Use the boto3-based approach
            response = generate_bedrock_response(messages)
            
            # Store the response
            last_conversation.append({"role": "assistant", "content": response})
            last_conversation = last_conversation[-2:]
            
            # Store in database (implement your store_db functions)
            # store_db.store_query_response_threat(request.user_id, request.query, response, "text")
            logger.info(f"chat response: {response}")
            
            return {"text": response, "type": "text"}
            
        except Exception as e:
            logger.error(f"Error: {str(e)}")
            final_answer = "Sorry, I couldn't retrieve the report."
            return {"text": final_answer, "type": "text"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "threat-analysis-api"}

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Cybersecurity Threat Analysis API",
        "version": "2.0.0",
        "endpoints": [
            "/threat - Simple threat analysis",
            "/threat_detailed - Detailed threat analysis",
            "/health - Health check"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)



# api.py - Fixed FastAPI endpoint that properly uses imported functions
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging
import json

# Import ALL functions from aws_functions.py
from aws_functions import generate_bedrock_response

# Setup logging
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(title="Cybersecurity Threat Analysis API", version="2.0.0")

class QueryRequest(BaseModel):
    user_id: str
    query: str

# Global conversation storage (in production, use a proper database)
last_conversation = []

# Simple conversation starters
simple_conversation_starters = {
    "hello": {"response": "Hello! How can I help you with cybersecurity analysis today?"},
    "hi": {"response": "Hi there! I'm here to help you analyze potential threats using VirusTotal."},
    "help": {"response": "I can help you analyze IPs, file hashes, URLs, and domains using VirusTotal. Just ask me to check something!"}
}

@app.post("/threat")
async def analyze_query(request: QueryRequest):
    """
    Analyze cybersecurity threats using AWS Bedrock and VirusTotal
    This endpoint uses the imported generate_bedrock_response function
    """
    global last_conversation
    
    print("client:", request)
    print("type(client):", type(request))
    
    lower_query = request.query.lower()
    logger.info(f"data: {request}")
    
    # Check for simple conversation starters
    if lower_query in simple_conversation_starters:
        predefined = simple_conversation_starters[lower_query]
        response = predefined["response"]
        return {"text": response, "type": "text"}
    
    else:
        # Update the conversation history
        last_conversation = last_conversation[-2:]  # Keep only the last conversation
        # Append the current query to the last conversation
        messages = last_conversation + [{"role": "user", "content": request.query}]
        last_conversation = last_conversation + [{"role": "user", "content": request.query}]
        
        try:
            # THIS IS WHERE WE USE THE IMPORTED FUNCTION
            response = generate_bedrock_response(messages)
            final_answer = response
            
            print(f"Generated response using imported function: {final_answer}")
            
            # Store the response in conversation history
            last_conversation.append({"role": "assistant", "content": final_answer})
            last_conversation = last_conversation[-2:]  # Keep only last 2 messages
            
            # Store in database (implement your store_db functions)
            # store_db.store_query_response_threat(request.user_id, request.query, final_answer, "text")
            logger.info(f"chat response: {final_answer}")
            
            return {"text": final_answer, "type": "text"}
            
        except Exception as e:
            logger.error(f"Error in analyze_query: {str(e)}")
            final_answer = "Sorry, I couldn't retrieve the report."
            # store_db.store_query_response_threat(request.user_id, request.query, final_answer, "text")
            logger.info(f"chat response: {final_answer}")
            return {"text": final_answer, "type": "text"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)