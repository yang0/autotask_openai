try:
    from autotask.nodes import Node, register_node
    from autotask.api_keys import get_api_key
    from autotask.llm import get_llm_config_by_id
except ImportError:
    from stub import Node, register_node, get_api_key

from typing import Dict, Any, Optional
from openai import OpenAI


@register_node
class TextGenerationNode(Node):
    NAME = "AI Text Generation"
    DESCRIPTION = "Generate text using OpenAI's language models with chat completion API"
    
    INPUTS = {
        "prompt": {
            "label": "Prompt",
            "description": "The text prompt to generate content from",
            "type": "STRING",
            "widget": "TEXTAREA",
            "required": True,
            "default": "Write a creative story"
        },
        "system_prompt": {
            "label": "System Prompt",
            "description": "Optional system message to set the behavior of the assistant",
            "type": "STRING",
            "widget": "TEXTAREA",
            "required": False,
            "default": "You are a helpful and creative assistant."
        },
        "max_tokens": {
            "label": "Maximum Length",
            "description": "Maximum number of tokens in the response",
            "type": "INTEGER",
            "required": False,
            "default": 1000,
            "minimum": 1,
            "maximum": 4000
        },
        "temperature": {
            "label": "Temperature",
            "description": "Controls randomness in the output (0.0-2.0)",
            "type": "FLOAT",
            "required": False,
            "default": 0.7,
            "minimum": 0.0,
            "maximum": 2.0
        },
        "llm_config_id": {
            "label": "AI Model",
            "description": "ID of the AI model configuration to use",
            "type": "STRING",
            "required": True,
            "widget": "LLM"
        }
    }
    
    OUTPUTS = {
        "generated_text": {
            "label": "Generated Text",
            "description": "The AI-generated text response",
            "type": "STRING"
        }
    }
    
    async def execute(self, node_inputs: Dict[str, Any], workflow_logger) -> Dict[str, Any]:
        try:
            # Get inputs
            prompt = node_inputs["prompt"]
            system_prompt = node_inputs.get("system_prompt", "You are a helpful and creative assistant.")
            max_tokens = node_inputs.get("max_tokens", 1000)
            temperature = node_inputs.get("temperature", 0.7)
            llm_config_id = node_inputs["llm_config_id"]
            
            # Get LLM configuration
            workflow_logger.info(f"Getting LLM configuration for ID: {llm_config_id}")
            llm_config = get_llm_config_by_id(llm_config_id)
            if not llm_config:
                raise ValueError(f"LLM configuration not found for ID: {llm_config_id}")
            
            params = llm_config.get_typed_parameters()
            if not params:
                raise ValueError("Failed to get typed parameters from LLM configuration")
            
            # Initialize OpenAI client
            workflow_logger.info("Initializing OpenAI client")
            client = OpenAI(
                api_key=params.api_key,
                base_url=params.base_url
            )
            
            # Prepare messages
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
            
            # Make API call
            workflow_logger.info("Sending request to AI model")
            completion = client.chat.completions.create(
                model=llm_config.llm_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            response = completion.choices[0].message.content
            workflow_logger.info("Successfully received AI response")
            
            return {
                "success": True,
                "generated_text": response
            }
            
        except Exception as e:
            error_msg = f"Text generation failed: {str(e)}"
            workflow_logger.error(error_msg)
            return {
                "success": False,
                "error_message": error_msg
            } 