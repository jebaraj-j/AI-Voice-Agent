import os
import speech_recognition as sr
import vertexai
from vertexai.preview.generative_models import GenerativeModel
from vertexai.preview.generative_models import SafetySetting, Tool, grounding
import simpleaudio as sa
import pyttsx3  # 🚀 Direct .wav audio generation

# 🔥 Initialize Vertex AI with Application Default Credentials
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
    "max_output_tokens": 6000,
    "temperature": 0.7,
    "top_p": 0.95,
    "seed": 0,
}

# 🚫 Safety Settings
safety_settings = [
    SafetySetting(
        category=SafetySetting.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        threshold=SafetySetting.HarmBlockThreshold.OFF
    ),
    SafetySetting(
        category=SafetySetting.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        threshold=SafetySetting.HarmBlockThreshold.OFF
    ),
    SafetySetting(
        category=SafetySetting.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        threshold=SafetySetting.HarmBlockThreshold.OFF
    ),
    SafetySetting(
        category=SafetySetting.HarmCategory.HARM_CATEGORY_HARASSMENT,
        threshold=SafetySetting.HarmBlockThreshold.OFF
    ),
]

# 🔎 Google Search Tool (optional)
tools = [
    Tool.from_google_search_retrieval(
        google_search_retrieval=grounding.GoogleSearchRetrieval()
    )
]

# 🎙️ Function to listen and recognize voice
def listen():
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        print("🎙️ Listening... (Speak clearly)")

        # 🔥 Noise reduction
        recognizer.adjust_for_ambient_noise(source, duration=1)  

        try:
            # ⏱️ Increased timeout and phrase time limit
            audio = recognizer.listen(source, timeout=15, phrase_time_limit=15)  
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

# 🔊 Function to convert text to speech and play it
def speak(text):
    output_file = "output.wav"

    # 🎯 Use pyttsx3 to generate `.wav` audio
    engine = pyttsx3.init()
    engine.save_to_file(text, output_file)
    engine.runAndWait()

    # 🎯 Play `.wav` audio with simpleaudio
    wave_obj = sa.WaveObject.from_wave_file(output_file)
    play_obj = wave_obj.play()
    play_obj.wait_done()

    # Clean up
    os.remove(output_file)

# 🚀 Main function
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
