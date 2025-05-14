# OpenAI Integration Plugin

This plugin provides integration with OpenAI's powerful AI models, enabling various AI capabilities in your workflows.

## Features

### Text Generation
- Generate creative text content using OpenAI's language models
- Support for system prompts and user prompts
- Adjustable parameters like maximum length and temperature
- Perfect for content creation, story writing, and text completion

### Image Recognition
- Analyze and describe image content using AI
- Support for both local images and image URLs
- Customizable prompts for specific analysis needs
- Ideal for image content understanding and description

### Image Generation (DALL-E)
- Create images from text descriptions
- Multiple size options (1024x1024, 1792x1024, 1024x1792)
- Quality settings (standard/HD)
- Style options (vivid/natural)
- Perfect for creating custom illustrations and artwork

### Speech to Text (Whisper)
- Convert audio files to text
- Support for multiple audio formats (mp3, mp4, mpeg, mpga, m4a, wav, webm)
- Optional language specification
- Guiding prompts for better transcription

### Text to Speech
- Convert text to natural-sounding speech
- Multiple voice options (alloy, echo, fable, onyx, nova, shimmer)
- High-quality audio output in MP3 format
- Perfect for creating voiceovers and audio content

## Requirements

- OpenAI API access (API key required)
- Supported AI models configured in your environment
- Python 3.7+
- Required Python packages:
  - openai
  - requests

## Usage

1. Configure your OpenAI API credentials in the AI Model settings
2. Add any of the provided nodes to your workflow
3. Configure the node parameters:
   - Select the appropriate AI model
   - Set input parameters (text, images, audio files, etc.)
   - Specify output locations where needed
4. Run your workflow

## Node Types

### TextGenerationNode
Text generation using chat completion API
```python
inputs = {
    "prompt": "Your text prompt",
    "system_prompt": "Optional system behavior setting",
    "max_tokens": 1000,
    "temperature": 0.7
}
```

### ImageRecognitionNode
Image analysis and description
```python
inputs = {
    "image_path": "path/to/image.jpg",
    "prompt": "What is in this image?"
}
```

### ImageGenerationNode
Create images from text descriptions
```python
inputs = {
    "prompt": "A detailed description of the image",
    "size": "1024x1024",
    "quality": "standard",
    "style": "vivid"
}
```

### SpeechToTextNode
Audio transcription
```python
inputs = {
    "audio_file": "path/to/audio.mp3",
    "language": "en",  # optional
    "prompt": "Optional transcription guide"
}
```

### TextToSpeechNode
Text to speech conversion
```python
inputs = {
    "text": "Text to convert to speech",
    "voice": "alloy",
    "output_file": "output.mp3"
}
```

## Error Handling

All nodes include comprehensive error handling and logging:
- Input validation
- API error handling
- File operation safety checks
- Detailed error messages in workflow logs

## Contributing

Feel free to contribute to this plugin by:
- Reporting issues
- Suggesting new features
- Submitting pull requests

## License

MIT License - feel free to use in your projects
