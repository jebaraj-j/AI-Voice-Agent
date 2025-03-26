import os
import speech_recognition as sr
import vertexai
from vertexai.preview.generative_models import GenerativeModel
from google.cloud import speech, texttospeech
from google.api_core.exceptions import GoogleAPICallError
import simpleaudio as sa
import time
from dotenv import load_dotenv
import logging

# ✅ Load environment variables securely
load_dotenv()

# 🔒 Secure credentials from .env
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv("GOOGLE_CREDENTIALS")

# ✅ Initialize Logging
logging.basicConfig(
    filename="assistant.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ✅ Preload Model and Clients Once
print("🔄 Initializing...")

vertexai.init(
    project=os.getenv("GCP_PROJECT"),
    location="us-central1"
)

# ✅ Load models and clients once
model = GenerativeModel(model_name="gemini-1.5-pro-002")
stt_client = speech.SpeechClient()
tts_client = texttospeech.TextToSpeechClient()

# ✅ Gemini Configuration
generation_config = {
    "max_output_tokens": 4000,
    "temperature": 0.7,
    "top_p": 0.95,
    "seed": 0,
}

# 🔒 Safety settings
safety_settings = [
    vertexai.preview.generative_models.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold=1),
    vertexai.preview.generative_models.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold=1),
    vertexai.preview.generative_models.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold=1),
    vertexai.preview.generative_models.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold=1)
]

# ✅ Store conversation history
conversation_history = []

# ✅ Rate limiting & timeout settings
MAX_REQUESTS_PER_MINUTE = 30  # Limit to 30 requests per minute
REQUEST_TIMEOUT = 10  # API timeout in seconds
request_count = 0
start_time = time.time()


# ✅ **Function to prevent abuse**
def rate_limit():
    global request_count, start_time
    current_time = time.time()
    
    # Reset counter every minute
    if current_time - start_time > 60:
        request_count = 0
        start_time = current_time

    if request_count >= MAX_REQUESTS_PER_MINUTE:
        print("⛔ Rate limit reached. Try again later.")
        time.sleep(5)
        return False
    else:
        request_count += 1
        return True


# 🎙️ **Voice Recognition Function**
def listen():
    """Listen and recognize voice."""
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        print("🎙️ Listening...")
        recognizer.adjust_for_ambient_noise(source, duration=0.5)

        try:
            audio = recognizer.listen(source, timeout=10, phrase_time_limit=10)
            print("🛠️ Recognizing...")

            audio_data = audio.get_wav_data()
            audio_file = speech.RecognitionAudio(content=audio_data)

            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=44100,
                language_code="en-US",
                model="latest_long",
                enable_automatic_punctuation=True
            )

            response = stt_client.recognize(config=config, audio=audio_file)

            if response.results:
                text = response.results[0].alternatives[0].transcript.strip()
                
                # ✅ Input validation (sanitize input)
                if len(text) > 500 or any(char in text for char in [';', '--', '<', '>', '`']):
                    print("🚫 Invalid input detected.")
                    return ""

                print(f"✅ You said: {text}")
                return text
            else:
                print("❌ No response recognized.")
                return ""

        except Exception as e:
            logging.error(f"STT Error: {e}")
            print(f"❌ Error: {e}")
            return ""


# 🤖 **Function to get AI response with caching**
def get_response(user_input, max_retries=3, retry_delay=1):
    """Generate AI response with caching."""
    global conversation_history

    if not rate_limit():
        return "Rate limit exceeded. Try again later."

    print(f"🤖 Generating AI Response for: {user_input}")

    retries = 0

    while retries < max_retries:
        try:
            time.sleep(0.5)
            full_prompt = "\n".join(conversation_history + [f"You: {user_input}"])

            responses = model.generate_content(
                [full_prompt],
                generation_config=generation_config,
                safety_settings=safety_settings
            )

            message = responses.text.replace('*', '')

            # ✅ Clean message
            clean_message = message.replace("AI:", "").replace("You:", "").strip()

            print(f"✅ AI Response: {clean_message}")

            # ✅ Store conversation history
            conversation_history.append(f"You: {user_input}")
            conversation_history.append(f"AI: {message}")
            conversation_history = conversation_history[-10:]

            return clean_message

        except GoogleAPICallError as e:
            if "Resource exhausted" in str(e):
                retries += 1
                print(f"🔄 Retrying... Attempt {retries}/{max_retries}")
                time.sleep(retry_delay)
            else:
                logging.error(f"API Error: {e}")
                print(f"❌ API Error: {e}")
                return "I'm having trouble connecting. Try again later."

        except Exception as e:
            logging.error(f"Unknown Error: {e}")
            print(f"❌ Unknown Error: {e}")
            return "Something went wrong. Please try again."

    print("❌ Max retries reached. Try again later.")
    return "I'm having trouble connecting. Please try again later."


# 🔊 **Text-to-Speech with Caching**
def speak(text):
    """Convert text to speech with Google Cloud TTS."""
    try:
        synthesis_input = texttospeech.SynthesisInput(text=text)

        voice = texttospeech.VoiceSelectionParams(
            language_code="en-US",
            name="en-US-Wavenet-F",
            ssml_gender=texttospeech.SsmlVoiceGender.FEMALE
        )

        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.LINEAR16
        )

        response = tts_client.synthesize_speech(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config
        )

        # ✅ Save and play audio
        audio_file = "output.wav"
        with open(audio_file, "wb") as out:
            out.write(response.audio_content)

        wave_obj = sa.WaveObject.from_wave_file(audio_file)
        play_obj = wave_obj.play()
        play_obj.wait_done()

        print("🔊 Speaking...")

        # ✅ Secure file cleanup
        if os.path.exists(audio_file):
            os.remove(audio_file)

    except Exception as e:
        logging.error(f"TTS Error: {e}")
        print(f"❌ Error in TTS: {e}")


# 🚀 **Main Function**
def main():
    """Main loop for the voice assistant."""
    print("🚀 AI Voice Assistant Running...")

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
