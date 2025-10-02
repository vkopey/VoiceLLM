import pyaudio
import wave
import whisper
import os, re

def record_audio():
    # Set parameters for recording
    FORMAT = pyaudio.paInt16  # 16-bit format
    CHANNELS = 1             # Mono audio
    RATE = 44100             # 44.1 kHz sample rate
    CHUNK = 1024             # Buffer size
    RECORD_SECONDS = 10       # Duration of recording
    OUTPUT_FILENAME = "output.wav"  # Output file name

    # Initialize PyAudio
    audio = pyaudio.PyAudio()

    # Open a stream for recording
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)

    print("Recording... 10s")
    frames = []

    # Record audio
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("Recording finished.")

    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    audio.terminate()

    # Save the recording as a .wav file
    with wave.open(OUTPUT_FILENAME, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))

    print(f"Audio saved to {OUTPUT_FILENAME}")

def run_whisper():
    model = whisper.load_model("turbo")
    result = model.transcribe("output.wav", language='uk', task="translate") # task="translate" need model "large"
    print(result["text"])
    return result["text"]

def run_llm(text):
    # Please install OpenAI SDK first: `pip3 install openai`

    from openai import OpenAI
    from APIkey import key

    client = OpenAI(api_key=key, base_url="https://api.deepseek.com")

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": text},
        ],
        stream=False
    )

    return response.choices[0].message.content

def run_ollama(text):
    from ollama import chat
    from ollama import ChatResponse

    response: ChatResponse = chat(model='qwen3:235b-a22b', messages=[
    {
        'role': 'user',
        'content': text+' Дай коротку відповідь. /nothink',
    },
    ])

    text=response.message.content
    return re.sub(r"<think>\s*</think>", "", text)

def run_tts(text):
    f=open("f:\\python-3.10.11-embed-amd64_TTS\\Scripts\\styletts2-ukrainian\\mytext.txt", "w")
    f.write(text)
    f.close()
    os.system('f:\\python-3.10.11-embed-amd64_TTS\\Scripts\\styletts2-ukrainian\\app3.bat')

record_audio()
text=run_whisper()
#text=run_llm(text)
text=run_ollama(text)
print(text)
run_tts(text)