# Installation Guide

## Prerequisites
- Python 3.9 or higher
- Ollama running locally on `http://localhost:11434/`

## Installation Steps

### 1. Create and activate virtual environment (if not already done)

**Windows (cmd.exe):**
```cmd
python -m venv myenv
myenv\Scripts\activate
```

**Windows (PowerShell):**
```powershell
python -m venv myenv
myenv\Scripts\Activate.ps1
```

**Linux/Mac:**
```bash
python -m venv myenv
source myenv/bin/activate
```

### 2. Install dependencies

```bash
pip install -e .
```

This will install all required packages including:
- LangChain and related packages
- Gradio for the UI
- SpeechRecognition for voice input
- httpx for API calls
- All other dependencies

### 3. Configure the agent

Edit `config.json` to set your preferences:
```json
{
  "model": {
    "base_url": "http://localhost:11434/",
    "name": "gpt-oss:20b",
    "temperature": 0.3
  },
  "query": "Get my location, check the weather, and give me the latest news"
}
```

### 4. Run the application

```bash
python app.py
```

The Gradio interface will launch and open in your browser at `http://127.0.0.1:7860`

## Features

- **Text Input**: Type your queries directly
- **Continuous Voice Recognition**:
  - Click "🎤 Start Voice" to activate continuous listening
  - Speak naturally - your queries are automatically captured and sent to the agent
  - Recognized queries are displayed with a 🎤 icon
  - Click "🛑 Stop Voice" to deactivate
  - No need to type "stop" - just speak your queries!
- **Connection Status**: Real-time Ollama connection monitoring
- **Tool Descriptions**: View available tools and their capabilities
- **Chat History**: See the conversation and agent responses
- **Quick Examples**: Pre-configured example queries for common tasks

## Troubleshooting

### Ollama Connection Error
- Ensure Ollama is running: `ollama serve`
- Check the base URL in `config.json` matches your Ollama server
- Verify the model name exists: `ollama list`

### Speech Recognition Issues
- Requires internet connection for Google Speech API
- Ensure microphone permissions are granted
- Check audio input device is working

### Module Not Found Errors
- Activate your virtual environment
- Run `pip install -e .` again
- Check Python version: `python --version` (should be 3.9+)
