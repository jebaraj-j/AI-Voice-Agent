import os
import speech_recognition as sr
import vertexai
from vertexai.preview.generative_models import GenerativeModel
from google.cloud import speech
import simpleaudio as sa
from gtts import gTTS
from pydub import AudioSegment  # For MP3 to WAV conversion

# ✅ Set the service account key
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "voice_ai_key.json"

# ✅ Initialize Vertex AI (Gemini API)
vertexai.init(
    project="abstract-arbor-454701-m0",  # Replace with your GCP project ID
    location="us-central1"
)
print("✅ Gemini API connected successfully!")

# ✅ Configure the Gemini model
model = GenerativeModel(
    model_name="gemini-1.5-pro-002"
)

# 🎯 Generation Configuration
generation_config = {
    "max_output_tokens": 4000,
    "temperature": 0.7,
    "top_p": 0.95,
    "seed": 0,
}

# 🔒 Safety settings
safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": 1},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": 1},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": 1},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": 1},
]

# 🎙️ **Function to listen and recognize voice using Google Cloud STT**
def listen():
    client = speech.SpeechClient()

    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("🎙️ Listening... (Speak clearly)")
        recognizer.adjust_for_ambient_noise(source, duration=1)

        try:
            audio = recognizer.listen(source, timeout=30, phrase_time_limit=30)
            print("🛠️ Recognizing...")

            # Convert audio to binary format
            audio_data = audio.get_wav_data()

            # Configure STT request
            audio_file = speech.RecognitionAudio(content=audio_data)
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=16000,
                language_code="en-US"
            )

            # Send the request
            response = client.recognize(config=config, audio=audio_file)

            for result in response.results:
                text = result.alternatives[0].transcript
                print(f"✅ You said: {text}")
                return text

        except sr.UnknownValueError:
            print("❌ Could not understand the audio.")
            return ""

        except sr.RequestError:
            print("❌ Could not request results.")
            return ""

        except sr.WaitTimeoutError:
            print("⏱️ No speech detected. Try again.")
            return ""

# 🤖 **Function to generate AI response using Gemini API**
def get_response(user_input):
    print(f"🤖 Generating AI Response for: {user_input}")

    responses = model.generate_content(
        [user_input],
        generation_config=generation_config,
        safety_settings=safety_settings
    )

    message = responses.text.replace('*', '')  # ✅ Remove all asterisks
    print(f"✅ AI Response: {message}")
    return message

# 🔊 **Function to convert text to speech and play it**
def speak(text):
    # Convert text to speech using gTTS
    tts = gTTS(text=text, lang='en')

    # Save as MP3 first
    mp3_file = "output.mp3"
    tts.save(mp3_file)

    # Convert MP3 to WAV
    wav_file = "output.wav"
    sound = AudioSegment.from_mp3(mp3_file)
    sound.export(wav_file, format="wav")

    # Play the WAV file
    wave_obj = sa.WaveObject.from_wave_file(wav_file)
    play_obj = wave_obj.play()
    play_obj.wait_done()

    # Clean up
    os.remove(mp3_file)
    os.remove(wav_file)

# 🚀 **Main function**
def main():
    print("🚀 Voice AI Assistant Running...")

    while True:
        user_input = listen()

        if "exit" in user_input.lower():
            print("👋 Exiting...")
            break

        if user_input:
            response = get_response(user_input)
            speak(response)

# ✅ Run the assistant
if __name__ == "__main__":
    main()
