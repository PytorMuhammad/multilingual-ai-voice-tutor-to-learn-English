![Multilingual AI Voice Tutor Banner](https://img.shields.io/badge/Multilingual%20AI%20Voice%20Tutor-%F0%9F%8E%99%EF%B8%8F%20%F0%9F%8C%90%20%F0%9F%A4%96-blueviolet?style=for-the-badge&logo=python)

# 🎯 **Multilingual AI Voice Tutor** 🎯

**✨ Welcome to the Multilingual AI Voice Tutor!** This cutting-edge application is your personal language coach, designed to help Urdu speakers master English with ease. Leveraging the power of AI, it offers an immersive, accent-free learning experience with support for multiple Text-to-Speech (TTS) providers, including **ElevenLabs** (best for accent bleeding). Whether you're practicing pronunciation or learning new vocabulary, this tutor has got you covered! 🚀

---

## 🎉 **Key Features** 🎉

- 🎤 **Multiple TTS Providers**: Choose from **ElevenLabs** 🔥 for top-notch speech synthesis.
- 🗣️ **Speech Recognition**: Powered by **OpenAI's Whisper** for precise Urdu and English transcription.
- 🔊 **Audio Processing**: Enhances clarity with amplification, noise reduction, and normalization.
- 🌐 **Interactive Interface**: Built with **Streamlit** for a seamless, web-based experience.

---

## 🎥 Demo Video

Watch the Multilingual AI Voice Tutor in action:

https://github.com/PytorMuhammad/multilingual-ai-voice-tutor-to-learn-English/blob/main/assets/VID-20250608-WA0012.mp4

<video width="100%" controls>
  <source src="./assets/VID-20250608-WA0012.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

---

## 🛠️ **Technologies Used** 🛠️

- 🐍 **Python 3.11**
- 🌟 **Streamlit** for the web interface
- 🎙️ **Whisper** for speech recognition
- 🤖 **OpenAI API** for NLP
- 🔊 **ElevenLabs API** for TTS
- 🎧 **Audio Processing Libraries**: librosa, pydub, noisereduce, scipy, sounddevice, soundfile

---

## 📂 **Project Structure** 📂

| File              | Description                           |
|-------------------|---------------------------------------|
| `app.py`          | 🚀 Application entry point, runs Streamlit. |
| `tutor_app.py`    | 🧠 Core logic, UI, and processing pipelines. |
| `Dockerfile`      | 🐳 For containerizing the application. |
| `requirements.txt`| 📦 Lists Python dependencies. |

---

## 🛠️ **Setup and Installation** 🛠️

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/PytorMuhammad/multilingual-ai-voice-tutor-to-learn-English.git
   cd multilingual-ai-voice-tutor-to-learn-English
   ```

2. **Install Dependencies**:
   Ensure **Python 3.11** is installed, then run:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set Up API Keys**:
   Configure your API keys in the Streamlit sidebar:
   - 🔑 **OpenAI** (for Whisper, ChatGPT)
   - 🔑 **ElevenLabs** (for TTS)

4. **Run the Application**:
   Launch the app with:
   ```bash
   python -u app.py
   ```
   Access it at `http://localhost:8501`.
   ```
---

## 🎓 **Usage** 🎓

### 🎤 **Voice Input**
1. Select "Voice" input in the sidebar.
2. Click "🔴 START RECORDING", speak clearly in Urdu or English, then "⏹️ STOP RECORDING".
3. Download the recording and upload it for processing.
4. Receive a transcribed response and audio output. 🎧

### 📝 **Text Input**
1. Select "Text" input in the sidebar.
2. Enter text with Urdu and English, or use the demo scenarios.
3. Click "🚀 Process Text" to generate a response and audio. 📄

### 🔊 **TTS Selection**
- Choose your preferred TTS provider (**ElevenLabs**) in the sidebar and adjust settings for an accent-free experience. 🎵

---

## ⚠️ **Known Issues** ⚠️

- 🔑 **API Dependency**: Requires valid API keys for full functionality.
- 🎧 **Audio Variability**: Quality may vary based on input and TTS provider.
- ⏳ **Processing Latency**: May vary depending on input complexity and provider.

---

## 🤝 **Contributing** 🤝

We welcome contributions! To get involved:
- 🐛 Report issues or suggest features via GitHub Issues.
- 💡 Submit pull requests with improvements.

---

**📅 Today's date and time: 12:48 PM PKT on Thursday, June 05, 2025**
