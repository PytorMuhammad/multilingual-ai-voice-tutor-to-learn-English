Multilingual AI Voice Tutor
Multilingual AI Voice Tutor is an interactive application designed to assist Urdu speakers in learning English. It integrates advanced AI technologies for speech recognition, natural language processing, and text-to-speech synthesis, delivering an immersive and accent-free learning experience. The app supports multiple TTS providers, including ElevenLabs, OpenAI, and Azure.
Key Features

Multiple TTS Providers: Select from ElevenLabs, OpenAI, or Azure for high-quality speech synthesis.
Speech Recognition: Powered by OpenAI's Whisper for accurate Urdu and English transcription.
Intelligent Language Tagging: Responses are tagged with [ur] for Urdu and [en] for English.
Audio Processing: Enhances clarity with amplification, noise reduction, and normalization.
Interactive Interface: Built with Streamlit for a seamless web-based experience.

Technologies Used

Python 3.11
Streamlit for the web interface
Whisper for speech recognition
OpenAI API for NLP and TTS
ElevenLabs API for TTS
Azure Cognitive Services for TTS
Audio Processing Libraries: librosa, pydub, noisereduce, scipy, sounddevice, soundfile

Project Structure

app.py: Application entry point, configures and runs Streamlit.
tutor_app.py: Core logic, including UI and processing pipelines.
Dockerfile: For containerizing the application.
requirements.txt: Lists Python dependencies.

Setup and Installation

Clone the Repository:
git clone https://github.com/your-repo/multilingual-ai-voice-tutor.git
cd multilingual-ai-voice-tutor


Install Dependencies:Ensure Python 3.11 is installed, then run:
pip install -r requirements.txt


Set Up API Keys:Configure API keys in the Streamlit sidebar for:

OpenAI (Whisper, ChatGPT, TTS)
ElevenLabs (TTS)
Azure Cognitive Services (TTS)


Run the Application:Launch the app with:
streamlit run app.py

Access it at http://localhost:8501.


Usage
Voice Input

Select "Voice" input in the sidebar.
Click "🔴 START RECORDING", speak in Urdu or English, then "⏹️ STOP RECORDING".
Download the recording and upload it for processing.
Receive a transcribed response and audio output.

Text Input

Select "Text" input in the sidebar.
Enter text with [ur] for Urdu and [en] for English, or use demo scenarios.
Click "🚀 Process Text" for a response and audio.

TTS Selection

Choose a TTS provider (ElevenLabs, OpenAI, Azure) in the sidebar and adjust settings.

Known Issues

API Dependency: Requires valid API keys for full functionality.
Audio Variability: Quality depends on input and TTS provider.
Processing Latency: May vary based on input complexity and provider.

Contributing
Contributions are welcome! To contribute:

Report issues or suggest features via GitHub Issues.
Submit pull requests with improvements.

