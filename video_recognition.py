try:
    from autotask.nodes import Node, register_node
    from autotask.api_keys import get_api_key
    from autotask.llm import get_llm_config_by_id, LLMConfig
except ImportError:
    from stub import Node, register_node

import os
from typing import Dict, Any, List, Optional
from openai import OpenAI
import base64


@register_node
class VideoRecognitionNode(Node):
    NAME = "AI Video Recognition"
    DESCRIPTION = "Use AI to analyze and describe video content using multiple frames"
    CATEGORY = "AI/ML"
    MAINTAINER = "AutoTask Team"
    ICON = "🎥"

    INPUTS = {
        "img1": {
            "label": "Image 1",
            "description": "First frame/image of the video",
            "type": "STRING",
            "widget": "FILE",
            "required": True,
            "default": ""
        },
        "img2": {
            "label": "Image 2",
            "description": "Second frame/image of the video",
            "type": "STRING",
            "widget": "FILE",
            "required": False,
            "default": ""
        },
        "img3": {
            "label": "Image 3",
            "description": "Third frame/image of the video",
            "type": "STRING",
            "widget": "FILE",
            "required": False,
            "default": ""
        },
        "img4": {
            "label": "Image 4",
            "description": "Fourth frame/image of the video",
            "type": "STRING",
            "widget": "FILE",
            "required": False,
            "default": ""
        },
        "img5": {
            "label": "Image 5",
            "description": "Fifth frame/image of the video",
            "type": "STRING",
            "widget": "FILE",
            "required": False,
            "default": ""
        },
        "img6": {
            "label": "Image 6",
            "description": "Sixth frame/image of the video",
            "type": "STRING",
            "widget": "FILE",
            "required": False,
            "default": ""
        },
        "img7": {
            "label": "Image 7",
            "description": "Seventh frame/image of the video",
            "type": "STRING",
            "widget": "FILE",
            "required": False,
            "default": ""
        },
        "img8": {
            "label": "Image 8",
            "description": "Eighth frame/image of the video",
            "type": "STRING",
            "widget": "FILE",
            "required": False,
            "default": ""
        },
        "prompt": {
            "label": "Analysis Prompt",
            "description": "Question or instruction for analyzing the video frames",
            "type": "STRING",
            "required": True,
            "default": "Describe what happens in this video sequence"
        },
        "llm_config_id": {
            "label": "LLM Configuration",
            "description": "ID of the LLM configuration to use",
            "type": "STRING",
            "required": True,
            "widget": "LLM",
            "default": ""
        }
    }

    OUTPUTS = {
        "description": {
            "label": "Video Description",
            "description": "AI-generated description or analysis of the video sequence",
            "type": "STRING"
        },
        "success": {
            "label": "Success",
            "description": "Whether the operation was successful",
            "type": "BOOLEAN"
        },
        "error_message": {
            "label": "Error Message",
            "description": "Error message if the operation failed",
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

    def _get_valid_image_paths(self, node_inputs: Dict[str, str]) -> List[str]:
        """Get all valid image paths from inputs."""
        valid_images = []
        for i in range(1, 9):
            img_key = f"img{i}"
            if img_key in node_inputs and node_inputs[img_key]:
                valid_images.append(node_inputs[img_key])
        return valid_images

    async def execute(self, node_inputs: Dict[str, str], workflow_logger) -> Dict[str, Any]:
        try:
            image_paths = self._get_valid_image_paths(node_inputs)
            if not image_paths:
                raise ValueError("At least one image path must be provided")

            prompt = node_inputs["prompt"]
            llm_config_id = node_inputs["llm_config_id"]

            workflow_logger.info(f"Getting LLM configuration for ID: {llm_config_id}")
            llm_config: Optional[LLMConfig] = get_llm_config_by_id(llm_config_id)
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

            # Prepare message content with video type
            image_urls = []
            for image_path in image_paths:
                image_url = self._get_image_url(image_path)
                image_urls.append(image_url)

            content = [
                {
                    "type": "video",
                    "video": image_urls
                },
                {
                    "type": "text",
                    "text": prompt
                }
            ]

            workflow_logger.info("Sending request to AI model")
            completion = client.chat.completions.create(
                model=llm_config.llm_name,
                messages=[{
                    "role": "user",
                    "content": content
                }]
            )

            response = completion.choices[0].message.content
            workflow_logger.info("Successfully received AI analysis")

            return {
                "success": True,
                "description": response,
                "error_message": ""
            }

        except Exception as e:
            error_msg = f"Video recognition failed: {str(e)}"
            workflow_logger.error(error_msg)
            return {
                "success": False,
                "description": "",
                "error_message": error_msg
            }
