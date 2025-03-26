import os
import speech_recognition as sr
import vertexai
from vertexai.preview.generative_models import GenerativeModel
from google.cloud import speech, texttospeech
from google.api_core.exceptions import GoogleAPICallError
import simpleaudio as sa
import time

# ✅ Preload Model and Clients Once (Faster Startup)
print("🔄 Initializing...")

# 🔥 Move initialization outside main loop for faster startup
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "voice_ai_key.json"

# ✅ Initialize Vertex AI once
vertexai.init(
    project="abstract-arbor-454701-m0",  
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


# 🎙️ **Voice Recognition Function**
def listen():
    """Listen and recognize voice."""
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        print("🎙️ Listening...")
        recognizer.adjust_for_ambient_noise(source, duration=0.5)  # Reduced noise adjustment time

        try:
            audio = recognizer.listen(source, timeout=10, phrase_time_limit=10)
            print("🛠️ Recognizing...")

            # Convert to WAV format
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
                text = response.results[0].alternatives[0].transcript
                print(f"✅ You said: {text}")
                return text
            else:
                print("❌ No response recognized.")
                return ""

        except Exception as e:
            print(f"❌ Error: {e}")
            return ""


# 🤖 **Function to get AI response with caching**
def get_response(user_input, max_retries=3, retry_delay=1):
    """Generate AI response with caching."""
    global conversation_history
    print(f"🤖 Generating AI Response for: {user_input}")

    retries = 0

    while retries < max_retries:
        try:
            time.sleep(0.5)  # Reduced delay
            full_prompt = "\n".join(conversation_history + [f"You: {user_input}"])

            responses = model.generate_content(
                [full_prompt],
                generation_config=generation_config,
                safety_settings=safety_settings
            )

            message = responses.text.replace('*', '')

            # ✅ Clean message to remove "AI:" and "You:" before speaking
            clean_message = message.replace("AI:", "").replace("You:", "").strip()

            print(f"✅ AI Response: {clean_message}")

            # ✅ Store conversation history (keep "You:" and "AI:" for context)
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
                print(f"❌ API Error: {e}")
                return "I'm having trouble connecting. Try again later."

        except Exception as e:
            print(f"❌ Unknown Error: {e}")
            return "Something went wrong. Please try again."

    print("❌ Max retries reached. Try again later.")
    return "I'm having trouble connecting. Please try again later."


# 🔊 **Text-to-Speech with Caching**
def speak(text):
    """Convert text to speech with Google Cloud TTS (faster playback)."""
    try:
        synthesis_input = texttospeech.SynthesisInput(text=text)

        voice = texttospeech.VoiceSelectionParams(
            language_code="en-US",
            name="en-US-Wavenet-F",      # ✅ Human-like female voice
            ssml_gender=texttospeech.SsmlVoiceGender.FEMALE
        )

        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.LINEAR16
        )

        # ✅ Synthesize speech
        response = tts_client.synthesize_speech(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config
        )

        # ✅ Save audio to file
        with open("output.wav", "wb") as out:
            out.write(response.audio_content)

        # ✅ Play audio faster
        wave_obj = sa.WaveObject.from_wave_file("output.wav")
        play_obj = wave_obj.play()
        play_obj.wait_done()

        print("🔊 Speaking...")

        # ✅ Clean up
        os.remove("output.wav")

    except Exception as e:
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
