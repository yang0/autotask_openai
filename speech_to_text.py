try:
    from autotask.nodes import Node, register_node
    from autotask.api_keys import get_api_key
    from autotask.llm import get_llm_config_by_id
except ImportError:
    from stub import Node, register_node, get_api_key

from typing import Dict, Any, Optional
from openai import OpenAI


@register_node
class SpeechToTextNode(Node):
    NAME = "Speech to Text"
    DESCRIPTION = "Convert audio to text using OpenAI's Whisper model"
    
    INPUTS = {
        "audio_file": {
            "label": "Audio File",
            "description": "Audio file to transcribe (supports mp3, mp4, mpeg, mpga, m4a, wav, webm)",
            "type": "STRING",
            "widget": "FILE",
            "required": True
        },
        "language": {
            "label": "Language",
            "description": "Language of the audio (optional, auto-detected if not specified)",
            "type": "STRING",
            "required": False,
            "default": ""
        },
        "prompt": {
            "label": "Prompt",
            "description": "Optional text to guide the model's style or continue a previous transcription",
            "type": "STRING",
            "widget": "TEXTAREA",
            "required": False,
            "default": ""
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
        "transcription": {
            "label": "Transcribed Text",
            "description": "The transcribed text from the audio file",
            "type": "STRING"
        }
    }
    
    async def execute(self, node_inputs: Dict[str, Any], workflow_logger) -> Dict[str, Any]:
        try:
            # Get inputs
            audio_file = node_inputs["audio_file"]
            language = node_inputs.get("language", "")
            prompt = node_inputs.get("prompt", "")
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
            
            # Prepare API call parameters
            api_params = {
                "model": llm_config.llm_name,
                "file": open(audio_file, "rb"),
            }
            
            if language:
                api_params["language"] = language
            if prompt:
                api_params["prompt"] = prompt
            
            # Make API call
            workflow_logger.info("Sending request to Whisper model")
            transcription = client.audio.transcriptions.create(**api_params)
            
            workflow_logger.info("Successfully received transcription")
            
            return {
                "success": True,
                "transcription": transcription.text
            }
            
        except Exception as e:
            error_msg = f"Speech to text conversion failed: {str(e)}"
            workflow_logger.error(error_msg)
            return {
                "success": False,
                "error_message": error_msg
            } 