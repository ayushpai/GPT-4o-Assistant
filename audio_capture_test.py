import sounddevice as sd
import numpy as np
import soundfile as sf
import speech_recognition as sr


def detect_and_record_audio(
    threshold=0.03, silence_duration=3, record_duration=5, samplerate=44100, channels=1
):
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
    sf.write("output.wav", audio_data, samplerate)
    print("Audio saved as output.wav")


detect_and_record_audio()
