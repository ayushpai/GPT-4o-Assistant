# Interactive Assistant Using GPT-4o

This repository contains the code for an advanced interactive assistant powered by OpenAI's newest model, GPT-4o. The assistant leverages multiple inputs, including screenshots and audio, to provide contextual and accurate responses to user queries. It integrates with a document database to ensure responses are based on relevant context from provided documents.

### Key Features

- **Screenshot Capture and Encoding**: Utilizes PyAutoGUI to capture screenshots and encode them in base64 format for input to the model.
- **Audio Detection and Transcription**: Detects and records audio using `sounddevice`, processes it with `Whisper`, and transcribes it to text.
- **Contextual Responses**: Employs document embeddings and similarity search with LangChain and SingleStoreDB to find relevant context for generating responses.
- **Text-to-Speech Output**: Converts the model's responses into speech using OpenAI's TTS capabilities.

### Technologies Used

- **OpenAI GPT-4o**: For generating responses.
- **OpenCV**: For image processing.
- **Whisper**: For audio transcription.
- **Sounddevice and Soundfile**: For audio handling.
- **Playsound**: For audio playback.
- **PyAutoGUI**: For screenshot capture.
- **LangChain**: For document processing and embedding.
- **SingleStoreDB**: For document storage and similarity search.

### How It Works

1. **Capture Screenshot**: The assistant captures a screenshot of the current screen.
2. **Record Audio**: It listens for user speech, records the audio, and transcribes it into text.
3. **Context Retrieval**: Searches for relevant context in a document database using similarity search.
4. **Generate Response**: Sends the transcribed text, captured screenshot, and relevant context to the GPT-4o model to generate a response.
5. **Text-to-Speech**: Converts the generated response to speech and plays it back to the user.

### Setup Instructions

1. Clone the repository.
2. Install the required dependencies:
   ```bash
   pip install openai opencv-python-headless sounddevice numpy soundfile speechrecognition whisper playsound pyautogui langchain_community singlestoredb
   ```
3. Set your OpenAI API key in the environment variable `OPENAI_API_KEY`.
4. Set your SingleStoreDB URL in the environment variable `SINGLESTOREDB_URL`.
5. Place your documents (e.g., `pytorch_docs.txt`) in the same directory.
6. Run the main script:
   ```bash
   python computer_assistant.py or python assistant.py
   ```

### Future Enhancements

- Integrate additional sensors and input methods.
- Improve audio quality and handling.
- Extend the assistant's capabilities for different use cases and domains.

Feel free to contribute and enhance this interactive assistant!
