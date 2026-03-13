"""
Morning Brief Agent UI - Gradio Interface

A web-based UI for the Morning Brief Agent workflow with voice and text input modes.
"""

import asyncio
import json
import os
from datetime import datetime

import gradio as gr
import httpx

from brief_workflow import agent_executor, TOOLS, config

# Try to import speech recognition
try:
    import speech_recognition as sr
    SPEECH_AVAILABLE = True
except ImportError:
    SPEECH_AVAILABLE = False
    print("Warning: speech_recognition not installed. Voice input will not work.")


def get_tool_descriptions() -> list:
    """Get formatted descriptions of available tools."""
    descriptions = []
    for tool in TOOLS:
        descriptions.append(f"**{tool.name}**: {tool.description}")
    return descriptions


def check_ollama_connection() -> tuple[bool, str]:
    """Check if Ollama server is running and accessible."""
    try:
        base_url = config["model"]["base_url"].rstrip("/")
        with httpx.Client() as client:
            response = client.get(f"{base_url}/api/tags", timeout=5.0)
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m.get("name", "unknown") for m in models]
                return True, f"Connected. Available models: {', '.join(model_names) if model_names else 'None'}"
            else:
                return False, f"Ollama server returned status {response.status_code}"
    except httpx.ConnectError:
        return False, "Cannot connect to Ollama server. Is it running?"
    except Exception as e:
        return False, f"Error checking Ollama: {str(e)}"


async def run_agent_async(query: str) -> str:
    """Run the agent asynchronously and return the result."""
    if not query.strip():
        return "Please enter a query."
    
    # Check connection first
    connected, status = check_ollama_connection()
    if not connected:
        return f"**Connection Error:** {status}\n\nPlease make sure Ollama is running on `{config['model']['base_url']}`"
    
    try:
        # Run the agent
        result = await agent_executor.ainvoke({"input": query})
        
        # Extract the output
        output = result.get("output", "No output generated")
        
        return output
        
    except httpx.ConnectError:
        return f"**Connection Error:** Cannot connect to Ollama server. Please make sure it's running on `{config['model']['base_url']}`"
    except Exception as e:
        error_msg = str(e)
        if "connection" in error_msg.lower() or "connect" in error_msg.lower():
            return f"**Connection Error:** {error_msg}\n\nPlease make sure Ollama is running on `{config['model']['base_url']}`"
        return f"**Error:** {error_msg}"


def run_agent(query: str) -> str:
    """Synchronous wrapper for the async agent execution."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        return loop.run_until_complete(run_agent_async(query))
    finally:
        loop.close()


def transcribe_audio(audio_path: str) -> str:
    """Transcribe audio file to text using speech recognition."""
    if not SPEECH_AVAILABLE:
        return "Speech recognition not available. Please install: pip install SpeechRecognition"
    
    if not audio_path:
        return ""
    
    try:
        recognizer = sr.Recognizer()
        with sr.AudioFile(audio_path) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)
            return text
    except sr.UnknownValueError:
        return "Could not understand audio"
    except sr.RequestError as e:
        return f"Speech recognition error: {e}"
    except Exception as e:
        return f"Error processing audio: {e}"


# Create the Gradio interface
with gr.Blocks(title="Morning Brief Agent") as app:
    
    # Header
    gr.HTML("""
        <div style="text-align: center; padding: 10px 0;">
            <h1 style="margin: 0; color: #1e40af;">Morning Brief Agent</h1>
            <p style="margin: 5px 0 0 0; color: #64748b;">Your AI-powered morning assistant</p>
        </div>
    """)
    
    # Check initial connection
    connected, status = check_ollama_connection()
    status_class = "status-connected" if connected else "status-disconnected"
    status_icon = "[OK]" if connected else "[ERROR]"
    
    # Main layout
    with gr.Row():
        # Left column - Main chat
        with gr.Column(scale=3):
            # Status indicator
            connection_status = gr.HTML(
                f"""<div style="padding: 10px; border-radius: 8px; background-color: {'#dcfce7' if connected else '#fee2e2'}; 
                border: 1px solid {'#86efac' if connected else '#fca5a5'}; color: {'#166534' if connected else '#991b1b'};">
                    <strong>{status_icon}</strong> {status}
                </div>"""
            )
            
            # Chat interface
            chatbot = gr.Chatbot(
                label="Conversation",
                height=400,
                show_label=False,
                container=True,
            )
            
            # Mode indicator
            mode_indicator = gr.HTML(
                """<div style="text-align: center; padding: 6px; background-color: #fef3c7; border-radius: 20px; 
                color: #92400e; font-weight: 600; margin-bottom: 10px;">
                    WRITE MODE - Type your query below
                </div>"""
            )
            
            # Input area
            with gr.Row():
                # Write mode input
                query_input = gr.Textbox(
                    label="Your Query",
                    placeholder="Type your query...",
                    lines=1,
                    show_label=False,
                    scale=4,
                    visible=True,
                    container=False,
                )
                
                # Voice mode input (compact)
                audio_input = gr.Audio(
                    sources=["microphone"],
                    type="filepath",
                    label="",
                    show_label=False,
                    scale=1,
                    visible=False,
                    waveform_options=gr.WaveformOptions(show_recording_waveform=False),
                    container=False,
                )
                
                submit_btn = gr.Button("Send", variant="primary", scale=1, visible=True)
            
            # Mode toggle and controls
            with gr.Row():
                write_mode_btn = gr.Button("Write", variant="primary", size="sm", scale=1)
                voice_mode_btn = gr.Button("Voice", variant="secondary", size="sm", scale=1)
                clear_btn = gr.Button("Clear", variant="secondary", size="sm", scale=1)
                refresh_btn = gr.Button("Refresh", variant="secondary", size="sm", scale=1)
            
            # Quick examples
            with gr.Row():
                example_btn1 = gr.Button("Weather & News", size="sm", variant="secondary")
                example_btn2 = gr.Button("My Location", size="sm", variant="secondary")
                example_btn3 = gr.Button("News", size="sm", variant="secondary")
        
        # Right column - Sidebar
        with gr.Column(scale=1):
            # Tools section
            gr.HTML("<h3 style='color: #1e40af; margin-bottom: 10px;'>Tools</h3>")
            
            tool_cards = []
            for tool in TOOLS:
                tool_cards.append(gr.HTML(f"""
                    <div style="background-color: #f1f5f9; border-radius: 6px; padding: 8px 12px; 
                    margin-bottom: 6px; border-left: 3px solid #3b82f6;">
                        <strong style="color: #1e293b;">{tool.name}</strong>
                        <br><small style="color: #64748b;">{tool.description}</small>
                    </div>
                """))
            
            # Configuration section (collapsible)
            with gr.Accordion("Configuration", open=False):
                model_info = gr.Textbox(
                    label="Model",
                    value=config["model"]["name"],
                    interactive=False,
                    container=False,
                )
                url_info = gr.Textbox(
                    label="URL",
                    value=config["model"]["base_url"],
                    interactive=False,
                    container=False,
                )
    
    # Event handlers
    def respond(message, chat_history):
        """Handle chat submission."""
        if not message.strip():
            return "", chat_history
        
        if chat_history is None:
            chat_history = []
        
        # Get agent response
        response = run_agent(message)
        
        # Add messages in Gradio 6.x format
        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": response})
        
        return "", chat_history
    
    def refresh_status():
        """Refresh the connection status."""
        connected, status = check_ollama_connection()
        status_icon = "[OK]" if connected else "[ERROR]"
        return f"""<div style="padding: 10px; border-radius: 8px; background-color: {'#dcfce7' if connected else '#fee2e2'}; 
        border: 1px solid {'#86efac' if connected else '#fca5a5'}; color: {'#166534' if connected else '#991b1b'};">
            <strong>{status_icon}</strong> {status}
        </div>"""
    
    def switch_to_write_mode():
        """Switch to write mode."""
        return (
            gr.Textbox(visible=True),    # query_input
            gr.Audio(visible=False),     # audio_input
            gr.Button(visible=True),     # submit_btn
            """<div style="text-align: center; padding: 6px; background-color: #fef3c7; border-radius: 20px; 
            color: #92400e; font-weight: 600; margin-bottom: 10px;">
                WRITE MODE - Type your query below
            </div>""",
            gr.Button(variant="primary"),    # write_mode_btn
            gr.Button(variant="secondary")   # voice_mode_btn
        )
    
    def switch_to_voice_mode():
        """Switch to voice mode."""
        return (
            gr.Textbox(visible=False),   # query_input
            gr.Audio(visible=True),      # audio_input
            gr.Button(visible=False),    # submit_btn
            """<div style="text-align: center; padding: 6px; background-color: #dbeafe; border-radius: 20px; 
            color: #1e40af; font-weight: 600; margin-bottom: 10px;">
                VOICE MODE - Click microphone and speak
            </div>""",
            gr.Button(variant="secondary"),  # write_mode_btn
            gr.Button(variant="primary")     # voice_mode_btn
        )
    
    def handle_voice_input(audio_file, chat_history):
        """Handle voice input - transcribe and send to agent."""
        if audio_file is None:
            return chat_history
        
        # Transcribe the audio
        transcribed = transcribe_audio(audio_file)
        
        if not transcribed or transcribed.startswith("Error") or transcribed.startswith("Could not"):
            # Show error in chat
            if chat_history is None:
                chat_history = []
            chat_history.append({"role": "assistant", "content": f"Error: {transcribed}"})
            return chat_history
        
        # Send to agent
        if chat_history is None:
            chat_history = []
        
        # Add user message with voice indicator
        chat_history.append({"role": "user", "content": f"Voice: \"{transcribed}\""})
        
        # Get agent response
        response = run_agent(transcribed)
        chat_history.append({"role": "assistant", "content": response})
        
        return chat_history
    
    # Write mode events
    submit_btn.click(
        respond,
        inputs=[query_input, chatbot],
        outputs=[query_input, chatbot]
    )
    
    query_input.submit(
        respond,
        inputs=[query_input, chatbot],
        outputs=[query_input, chatbot]
    )
    
    # Mode switching
    write_mode_btn.click(
        switch_to_write_mode,
        outputs=[query_input, audio_input, submit_btn, mode_indicator, write_mode_btn, voice_mode_btn]
    )
    
    voice_mode_btn.click(
        switch_to_voice_mode,
        outputs=[query_input, audio_input, submit_btn, mode_indicator, write_mode_btn, voice_mode_btn]
    )
    
    # Voice input handler
    audio_input.change(
        handle_voice_input,
        inputs=[audio_input, chatbot],
        outputs=[chatbot]
    )
    
    # Clear and refresh
    clear_btn.click(
        lambda: ([], ""),
        outputs=[chatbot, query_input]
    )
    
    refresh_btn.click(
        refresh_status,
        outputs=[connection_status]
    )
    
    # Example buttons
    example_btn1.click(
        lambda: "Give me the weather conditions and latest news today and save it under reports folder.",
        outputs=[query_input]
    )
    
    example_btn2.click(
        lambda: "What's my current location?",
        outputs=[query_input]
    )
    
    example_btn3.click(
        lambda: "Get me the latest news headlines.",
        outputs=[query_input]
    )


if __name__ == "__main__":
    # Create reports directory if it doesn't exist
    os.makedirs("reports", exist_ok=True)
    
    # Launch the main interface
    app.launch(
        server_name="127.0.0.1",
        server_port=None,  # Auto-find available port
        share=False,
        show_error=True,
    )
