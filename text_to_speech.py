try:
    from autotask.nodes import Node, register_node
    from autotask.api_keys import get_api_key
    from autotask.llm import get_llm_config_by_id
except ImportError:
    from stub import Node, register_node, get_api_key

import os
from typing import Dict, Any, Optional
from openai import OpenAI


@register_node
class TextToSpeechNode(Node):
    NAME = "Text to Speech"
    DESCRIPTION = "Convert text to natural-sounding speech using OpenAI's TTS models"
    
    INPUTS = {
        "text": {
            "label": "Text",
            "description": "The text to convert to speech",
            "type": "STRING",
            "widget": "TEXTAREA",
            "required": True
        },
        "voice": {
            "label": "Voice",
            "description": "The voice to use for the speech",
            "type": "STRING",
            "required": True,
            "default": "alloy",
            "choices": ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
        },
        "output_file": {
            "label": "Output File",
            "description": "Path to save the generated audio file (mp3 format)",
            "type": "STRING",
            "widget": "FILE",
            "required": True,
            "default": "output.mp3"
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
        "audio_path": {
            "label": "Audio File Path",
            "description": "Path to the generated audio file",
            "type": "STRING"
        }
    }
    
    async def execute(self, node_inputs: Dict[str, Any], workflow_logger) -> Dict[str, Any]:
        try:
            # Get inputs
            text = node_inputs["text"]
            voice = node_inputs["voice"]
            output_file = node_inputs["output_file"]
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
            
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
            
            # Make API call
            workflow_logger.info("Sending request to TTS model")
            response = client.audio.speech.create(
                model=llm_config.llm_name,
                voice=voice,
                input=text
            )
            
            # Save the audio file
            response.stream_to_file(output_file)
            workflow_logger.info(f"Successfully saved audio to {output_file}")
            
            return {
                "success": True,
                "audio_path": output_file
            }
            
        except Exception as e:
            error_msg = f"Text to speech conversion failed: {str(e)}"
            workflow_logger.error(error_msg)
            return {
                "success": False,
                "error_message": error_msg
            } 