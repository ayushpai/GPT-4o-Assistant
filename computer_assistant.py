import base64
import time
from openai import OpenAI
import cv2
import sounddevice as sd
import numpy as np
import soundfile as sf
import speech_recognition as sr
import whisper
from pathlib import Path
from playsound import playsound
import os
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores.singlestoredb import SingleStoreDB
import pyautogui

"""
Interactive Computer Assistant

"""


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def assistant(llm_input, llm_history, client, context):

    screenshot = pyautogui.screenshot()
    screenshot.save("screenshot.png")
    base64_image = encode_image("screenshot.png")
    
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "Only use the provided prompt text and image to answer question. If it is not in document, do not answer.",
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "User Question (If answer is not in context document say I don't know). Answer should be 3 sentences max. Provide no code in your answer: " + llm_input + "\nContext Document: " + context},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                ],
            },
            
        ],
        max_tokens=300,
    )

    response_text = response.choices[0].message.content
    return response_text


def detect_and_record_audio(threshold=0.03, silence_duration=3, record_duration=5, samplerate=44100, channels=1):
    recognizer = sr.Recognizer()
    print("Listening for speech...")

    started = False

    def callback(indata, frames, time, status):
        nonlocal started
        if np.any(indata > threshold):
            if not started:
                print("Starting recording...")
                started = True
                raise sd.CallbackAbort

    # Detect speech
    with sd.InputStream(callback=callback, channels=channels, samplerate=samplerate):
        while not started:
            sd.sleep(100)

    # Record for the specified duration after speech detection
    audio_data = sd.rec(
        int(record_duration * samplerate),
        samplerate=samplerate,
        channels=channels,
        dtype="float32",
    )
    sd.wait()  # Wait until the recording is finished
    sf.write("voice_input.wav", audio_data, samplerate)
    print("Audio saved as voice_input.wav")




def main():
    llm_history = []
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    os.environ["SINGLESTOREDB_URL"] = "ayush:Test1234@svc-3482219c-a389-4079-b18b-d50662524e8a-shared-dml.aws-virginia-6.svc.singlestore.com:3333/database_79fb0"

    # Load and process documents
    loader = TextLoader("pytorch_docs.txt")
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)

    # Generate embeddings and create a document search database
    embeddings = OpenAIEmbeddings(api_key=os.environ.get("OPENAI_API_KEY"))
    vectorDB = SingleStoreDB.from_documents(docs, embeddings, table_name="data")


    while True:
        detect_and_record_audio()
        model = whisper.load_model("base")
        result = model.transcribe("voice_input.wav")
        llm_input = result["text"]
        print(llm_input)

        docs = vectorDB.similarity_search(llm_input)
        context = docs[0].page_content
        
        llm_output = assistant(llm_input, llm_history, client, context)
        llm_history = llm_history + [{"role": "assistant", "content": llm_output}]
        print(llm_output)



        response = client.audio.speech.create(
            model="tts-1",
            voice="fable",
            input=llm_output,
        )

        
        response.stream_to_file("output.mp3")
        playsound("output.mp3")

        time.sleep(2)
        

if __name__ == "__main__":
    main()
