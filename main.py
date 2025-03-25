import os
import speech_recognition as sr
import vertexai
from vertexai.preview.generative_models import GenerativeModel
import simpleaudio as sa
from gtts import gTTS
import io
import wave

# ✅ Initialize Vertex AI
vertexai.init(
    project="abstract-arbor-454701-m0",  # Your GCP project ID
    location="us-central1"
)

# ✅ Configure the Gemini model
model = GenerativeModel(
    model_name="gemini-1.5-pro-002"  # Latest version of Gemini model
)

# 🎯 Generation Configuration
generation_config = {
    "max_output_tokens": 100,   # Shorter responses
    "temperature": 0.7,
    "top_p": 0.95,
    "seed": 0,
}

# 🎙️ **Function to listen and recognize voice**
def listen():
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        print("🎙️ Listening... (Speak clearly)")
        recognizer.adjust_for_ambient_noise(source, duration=1)  # Noise reduction
        try:
            audio = recognizer.listen(source, timeout=15, phrase_time_limit=15)  # Extended listening time
            print("🛠️ Recognizing...")
            text = recognizer.recognize_google(audio)
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

# 🤖 **Function to generate AI response**
def get_response(user_input):
    print(f"🤖 Generating AI Response for: {user_input}")
    
    responses = model.generate_content(
        [user_input],
        generation_config=generation_config
    )

    # Extract and return the response
    message = responses.text.strip().replace("*", "")  # Remove asterisks
    print(f"✅ AI Response: {message}")
    return message

# 🔊 **Function to convert text to WAV and play it**
def speak(text):
    # Convert text to speech using gTTS
    tts = gTTS(text=text, lang='en')

    # Save to in-memory bytes buffer
    with io.BytesIO() as buffer:
        tts.write_to_fp(buffer)
        buffer.seek(0)

        # Create a WAV file directly
        with wave.open("output.wav", "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(44100)
            wav_file.writeframes(buffer.read())

    # Play the WAV file
    wave_obj = sa.WaveObject.from_wave_file("output.wav")
    play_obj = wave_obj.play()
    play_obj.wait_done()

    # Clean up
    os.remove("output.wav")

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
