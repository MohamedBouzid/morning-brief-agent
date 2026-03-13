# Morning Brief Agent

A subscription-free AI agent that delivers personalized morning briefings with weather, news, and location data using local LLMs.

## 🌟 Features

- **Weather Information**: Get current weather conditions for any location
- **Latest News**: Fetch breaking news from reliable sources
- **Location Detection**: Automatically detect your location via IP
- **File Saving**: Save briefings to organized reports with timestamps
- **Parallel Processing**: Execute multiple tasks simultaneously for faster results (Not implemented yet)
- **Configurable**: Easy configuration through JSON config file
- **Local LLM**: Uses Ollama for privacy and cost-free operation
- **Async Operations**: Built with async/await for efficient I/O operations

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- [Ollama](https://ollama.ai/) installed and running
- A local LLM model (e.g., `llama3.1:8b`, `qwen2.5:7b`)

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd morning-brief-agent
   ```

2. **Install Python dependencies:**
   ```bash
   pip install -e .
   ```

3. **Install and start Ollama:**
   ```bash
   # Install Ollama (follow instructions at https://ollama.ai/)
   ollama serve

   # Pull a model (choose one that supports tool calling)
   ollama pull gpt-oss:20b
   ```

### Configuration

Edit `config.json` to customize your setup:

```json
{
  "model": {
    "name": "gpt-oss:20b",
    "base_url": "http://localhost:11434/",
    "temperature": 0.3
  },
  "query": "Give me the weather conditions and latest news today and save it under reports folder. The filename should be suffixed with the date time."
}
```

**Configuration Options:**
- `model.name`: The Ollama model to use (must support tool calling)
- `model.base_url`: Ollama server URL (default: http://localhost:11434/)
- `model.temperature`: Creativity level (0.0-1.0)
- `query`: The default query to execute

### Usage

**Run the morning brief:**
```bash
python brief_workflow.py
```

**Or use the installed script:**
```bash
brief
```

**Custom query:**
```python
from brief_workflow import main
import asyncio

async def custom_brief():
    # Your custom logic here
    pass

asyncio.run(custom_brief())
```

## 🏗️ Architecture

### Core Components

- **Tools**: Async functions decorated with `@tool` for specific tasks
  - `weather_tool`: Fetches weather data from Open-Meteo API
  - `news_tool`: Scrapes latest news from BBC
  - `get_location_tool`: Detects location via IP geolocation
  - `save_to_file`: Saves data to timestamped files

- **Agent**: LangChain tool-calling agent with parallel execution
- **Evaluator**: Custom performance evaluation system
- **Configuration**: JSON-based configuration management

### Project Structure

```
morning-brief-agent/
├── brief_workflow.py      # Main agent workflow
├── config.json           # Configuration file
├── pyproject.toml        # Python project configuration
├── package.json          # Node.js dependencies (MCP)
├── reports/              # Generated briefings
├── myenv/                # Python virtual environment
├── README.md            # This file
└── LICENSE              # MIT License
```

## 🔧 Development

### Adding New Tools

Create async functions decorated with `@tool`:

```python
@tool
async def my_new_tool(param: str) -> str:
    """Tool description."""
    # Your async logic here
    return "Result"
```

Add to the `TOOLS` list and the agent will automatically use it.

### Testing

```bash
# Install dev dependencies
pip install -e ".[dev]"
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [LangChain](https://github.com/langchain-ai/langchain) for the agent framework
- [Ollama](https://ollama.ai/) for local LLM support
- [Open-Meteo](https://open-meteo.com/) for weather data
- [BBC News](https://bbc.com/news) for news content

## 🔍 Troubleshooting

**Debug Mode:**
Enable verbose logging by setting `verbose=True` in the agent executor.

## 📞 Support

For issues and questions:
- Open an issue on GitHub
- Check the troubleshooting section
- Review the configuration options