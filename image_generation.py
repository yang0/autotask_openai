try:
    from autotask.nodes import Node, register_node
    from autotask.api_keys import get_api_key
    from autotask.llm import get_llm_config_by_id
except ImportError:
    from stub import Node, register_node, get_api_key

import os
from typing import Dict, Any, Optional
from openai import OpenAI
import requests


@register_node
class ImageGenerationNode(Node):
    NAME = "AI Image Generation"
    DESCRIPTION = "Generate images from text descriptions using OpenAI's DALL-E models"
    
    INPUTS = {
        "prompt": {
            "label": "Prompt",
            "description": "Text description of the image you want to generate",
            "type": "STRING",
            "widget": "TEXTAREA",
            "required": True
        },
        "size": {
            "label": "Image Size",
            "description": "The size of the generated image",
            "type": "STRING",
            "required": True,
            "default": "1024x1024",
            "choices": ["1024x1024", "1792x1024", "1024x1792"]
        },
        "quality": {
            "label": "Image Quality",
            "description": "The quality of the generated image",
            "type": "STRING",
            "required": False,
            "default": "standard",
            "choices": ["standard", "hd"]
        },
        "style": {
            "label": "Image Style",
            "description": "The style of the generated image",
            "type": "STRING",
            "required": False,
            "default": "vivid",
            "choices": ["vivid", "natural"]
        },
        "output_file": {
            "label": "Output File",
            "description": "Path to save the generated image",
            "type": "STRING",
            "widget": "FILE",
            "required": True,
            "default": "output.png"
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
        "image_path": {
            "label": "Image File Path",
            "description": "Path to the generated image file",
            "type": "STRING"
        }
    }
    
    def _download_image(self, image_url: str, output_path: str) -> None:
        """Download image from URL and save to file."""
        response = requests.get(image_url)
        response.raise_for_status()
        
        with open(output_path, 'wb') as f:
            f.write(response.content)
    
    async def execute(self, node_inputs: Dict[str, Any], workflow_logger) -> Dict[str, Any]:
        try:
            # Get inputs
            prompt = node_inputs["prompt"]
            size = node_inputs["size"]
            quality = node_inputs.get("quality", "standard")
            style = node_inputs.get("style", "vivid")
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
            workflow_logger.info("Sending request to DALL-E model")
            response = client.images.generate(
                model=llm_config.llm_name,
                prompt=prompt,
                size=size,
                quality=quality,
                style=style,
                n=1
            )
            
            # Download and save the image
            image_url = response.data[0].url
            self._download_image(image_url, output_file)
            workflow_logger.info(f"Successfully saved image to {output_file}")
            
            return {
                "success": True,
                "image_path": output_file
            }
            
        except Exception as e:
            error_msg = f"Image generation failed: {str(e)}"
            workflow_logger.error(error_msg)
            return {
                "success": False,
                "error_message": error_msg
            } 