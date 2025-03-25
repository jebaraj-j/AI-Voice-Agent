import os
import speech_recognition as sr
import vertexai
from vertexai.preview.generative_models import GenerativeModel
from google.cloud import speech
import pyttsx3
from vertexai.preview.generative_models import SafetySetting
from google.api_core.exceptions import GoogleAPICallError, RetryError
import time

# ✅ Set the service account key
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "voice_ai_key.json"

# ✅ Initialize Vertex AI (Gemini API)
vertexai.init(
    project="abstract-arbor-454701-m0",  # Your GCP project ID
    location="us-central1"
)
print("✅ Gemini API connected successfully!")

# ✅ Configure the Gemini model
model = GenerativeModel(model_name="gemini-1.5-pro-002")

# 🎯 Generation Configuration
generation_config = {
    "max_output_tokens": 4000,
    "temperature": 0.7,
    "top_p": 0.95,
    "seed": 0,
}

# 🔒 Safety settings
safety_settings = [
    SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold=1),
    SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold=1),
    SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold=1),
    SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold=1)
]

# ✅ Store conversation history
conversation_history = []


# 🎙️ **Function to listen and recognize voice**
def listen():
    """Listen and recognize voice using Google Cloud STT."""
    recognizer = sr.Recognizer()
    client = speech.SpeechClient()

    with sr.Microphone() as source:
        print("🎙️ Listening... (Speak clearly)")
        recognizer.adjust_for_ambient_noise(source, duration=1)  # Noise reduction

        try:
            audio = recognizer.listen(source, timeout=15, phrase_time_limit=15)
            print("🛠️ Recognizing...")

            # Convert to WAV format
            audio_data = audio.get_wav_data()

            # ✅ Google Cloud STT configuration (Improved accuracy)
            audio_file = speech.RecognitionAudio(content=audio_data)

            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=44100,   # Match sample rate
                language_code="en-US",      # English only
                model="latest_long",        # Use latest model for better accuracy
                enable_automatic_punctuation=True  # Better sentence formatting
            )

            response = client.recognize(config=config, audio=audio_file)

            if response.results:
                text = response.results[0].alternatives[0].transcript
                print(f"✅ You said: {text}")
                return text
            else:
                print("❌ No response recognized.")
                return ""

        except sr.UnknownValueError:
            print("❌ Could not understand the audio.")
            return ""

        except sr.RequestError:
            print("❌ Could not request results.")
            return ""

        except sr.WaitTimeoutError:
            print("⏱️ No speech detected. Try again.")
            return ""

        except Exception as e:
            print(f"❌ Error: {e}")
            return ""


# 🤖 **Function to generate AI response with retry and rate limiting**
def get_response(user_input, max_retries=3, retry_delay=2):
    """Generate AI response with Gemini API and retry on 429 errors."""
    global conversation_history
    print(f"🤖 Generating AI Response for: {user_input}")

    retries = 0

    while retries < max_retries:
        try:
            # ✅ Add rate-limiting delay
            time.sleep(1.5)

            # ✅ Include conversation history
            full_prompt = "\n".join(conversation_history + [f"You: {user_input}"])

            responses = model.generate_content(
                [full_prompt],
                generation_config=generation_config,
                safety_settings=safety_settings
            )

            message = responses.text.replace('*', '')
            print(f"✅ AI Response: {message}")

            # ✅ Store conversation history (last 10 exchanges)
            conversation_history.append(f"You: {user_input}")
            conversation_history.append(f"AI: {message}")
            conversation_history = conversation_history[-10:]

            return message

        except GoogleAPICallError as e:
            if "Resource exhausted" in str(e):
                retries += 1
                print(f"🔄 Retrying... Attempt {retries}/{max_retries}")
                time.sleep(retry_delay)
            else:
                print(f"❌ API Error: {e}")
                return "I'm having trouble connecting right now. Please try again later."

        except Exception as e:
            print(f"❌ Unknown Error: {e}")
            return "Something went wrong. Please try again."

    print("❌ Max retries reached. Try again later.")
    return "I'm having trouble connecting. Please try again later."


# 🔊 **Function to convert text to speech with standard voice**
def speak(text):
    """Convert text to speech with a consistent voice using pyttsx3."""
    try:
        engine = pyttsx3.init()

        # ✅ Set a consistent voice
        voices = engine.getProperty('voices')

        # Choose a specific voice for consistency
        engine.setProperty('voice', voices[1].id)  # Female voice (index may vary)

        # ✅ Set the speed and volume for standardization
        engine.setProperty('rate', 180)  # Normal speaking rate
        engine.setProperty('volume', 1.0)  # Max volume

        print("🔊 Speaking...")
        engine.say(text)
        engine.runAndWait()

    except Exception as e:
        print(f"❌ Error in TTS: {e}")


# 🚀 **Main function**
def main():
    print("🚀 Voice AI Assistant Running...")

    while True:
        user_input = listen()

        if user_input and "exit" in user_input.lower():
            print("👋 Exiting...")
            break

        if user_input:
            response = get_response(user_input)
            speak(response)


# ✅ Run the assistant
if __name__ == "__main__":
    main()
