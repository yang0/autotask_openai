try:
    from autotask.nodes import Node, register_node
    from autotask.api_keys import get_api_key
    from autotask.llm import get_llm_config_by_id
except ImportError:
    from stub import Node, register_node, get_api_key

import os
from typing import Dict, Any, Optional
from openai import OpenAI
import base64


@register_node
class ImageRecognitionNode(Node):
    NAME = "AI Image Recognition"
    DESCRIPTION = "Use AI to analyze and describe image content using various LLM models"

    INPUTS = {
        "image_path": {
            "label": "Image Path",
            "description": "Path to the image file to analyze",
            "type": "STRING",
            "widget": "FILE",
            "required": True
        },
        "prompt": {
            "label": "Analysis Prompt",
            "description": "Question or instruction for analyzing the image",
            "type": "STRING",
            "required": True,
            "default": "What is in this image?"
        },
        "llm_config_id": {
            "label": "LLM Configuration",
            "description": "ID of the LLM configuration to use",
            "type": "STRING",
            "required": True,
            "widget": "LLM",
        }
    }

    OUTPUTS = {
        "description": {
            "label": "Image Description",
            "description": "AI-generated description or analysis of the image",
            "type": "STRING"
        }
    }

    def _encode_image_to_base64(self, image_path: str) -> str:
        """Convert image to base64 string."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def _get_image_url(self, image_path: str) -> str:
        """Get image URL or base64 data URL."""
        if image_path.startswith(('http://', 'https://')):
            return image_path
        else:
            base64_image = self._encode_image_to_base64(image_path)
            return f"data:image/jpeg;base64,{base64_image}"

    async def execute(self, node_inputs: Dict[str, Any], workflow_logger) -> Dict[str, Any]:
        try:
            image_path = node_inputs["image_path"]
            prompt = node_inputs["prompt"]
            llm_config_id = node_inputs["llm_config_id"]

            workflow_logger.info(f"Getting LLM configuration for ID: {llm_config_id}")
            llm_config = get_llm_config_by_id(llm_config_id)
            if not llm_config:
                raise ValueError(f"LLM configuration not found for ID: {llm_config_id}")

            params = llm_config.get_typed_parameters()
            if not params:
                raise ValueError("Failed to get typed parameters from LLM configuration")

            workflow_logger.info("Initializing OpenAI client")
            client = OpenAI(
                api_key=params.api_key,
                base_url=params.base_url
            )

            image_url = self._get_image_url(image_path)
            
            workflow_logger.info("Sending request to AI model")
            completion = client.chat.completions.create(
                model=llm_config.llm_name,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_url}}
                    ]
                }]
            )

            response = completion.choices[0].message.content
            workflow_logger.info("Successfully received AI analysis")

            return {
                "success": True,
                "description": response
            }

        except Exception as e:
            error_msg = f"Image recognition failed: {str(e)}"
            workflow_logger.error(error_msg)
            return {
                "success": False,
                "error_message": error_msg
            } 