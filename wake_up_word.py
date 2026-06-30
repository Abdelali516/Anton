import pyaudio
import numpy as np
import subprocess
from openwakeword.model import Model
import time


SAMPLE_RATE = 16000
CHUNK = 1280

oww_model = Model()

def listen_for_wakeword():
    audio = pyaudio.PyAudio()
    stream = audio.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=CHUNK
    )


    while True:
        data = stream.read(CHUNK, exception_on_overflow=False)
        frame = np.frombuffer(data, dtype=np.int16)
        prediction = oww_model.predict(frame)


        for key, score in prediction.items():
            if score > 0.5:
                print("Wake word detected!")
                stream.stop_stream()
                stream.close()
                audio.terminate()
                return

while True:
    listen_for_wakeword()
    process = subprocess.Popen([
        "/home/abdelali/ollama-venv/bin/python",
        "/home/abdelali/Downloads/Anton.py/main.py"
    ])
    process.wait()
