import json
from flask import Flask, request, jsonify
from flask_cors import CORS
import pygame
import speech_recognition as sr
from googletrans import Translator
from gtts import gTTS
import os
import time
import google.generativeai as genai

genai.configure(api_key="AIzaSyCW3Zz4xkIiaj7YIFpml-KNK58KsJyekaA")
model = genai.GenerativeModel('gemini-1.5-flash')

app = Flask(__name__)
CORS(app)

# Supported languages
supported_languages = ['en', 'te', 'hi']
translator = Translator()

# Function to convert text to speech and play it
def speak(text, language):
    tts = gTTS(text=text, lang=language)
    audio_file = "temp.mp3"
    tts.save(audio_file)

    # Initialize pygame mixer
    pygame.mixer.init()
    pygame.mixer.music.load(audio_file)
    pygame.mixer.music.play()

    # Keep the program running until the sound is done playing
    while pygame.mixer.music.get_busy():
        time.sleep(0.1)

    # Stop the mixer and cleanup
    pygame.mixer.music.stop()
    pygame.mixer.quit()

# API route to handle voice input and generate a response
@app.route('/getResponse', methods=['GET'])
def getResponse():
    try:
        # Extract language and page from the request
        language = request.args.get('language', 'en')
        page = request.args.get('page')

        # Validate the language
        if language not in supported_languages:
            return jsonify({"error": "Unsupported language"}), 400

        # Generate the appropriate response based on the page
        if page == 'crop_disease':
            response_text = 'Welcome to the Crop Disease section. Please upload a valid image of your crop affected by disease.'
        elif page == 'recommendation_system':
            response_text = 'Click the Fetch Location Button if when you at your farm location or else enter your crop data manually. Now you will get recommendations here'
        elif page == 'yield_prediction':
            response_text = 'You can check the predicted yield for your crops here. Click on the respective crop to get their specific yield details. And, we will also provide suggestions based on your yield and crop data.'
        else:
            response_text = 'Welcome to the application.'

        # Translate the response text to the user's selected language
        try:
            translation = translator.translate(response_text, src='en', dest=language)
            response_translated = translation.text if translation else 'Translation failed'
        except Exception as e:
            print("Translation error:", e)
            return jsonify({"error": "Translation failed"}), 500

        print("Translated text:", response_translated)

        # Convert the response to speech
        speak(response_translated, language)

        # Return the response as JSON
        return jsonify({
            'message': response_translated
        }), 200
    except Exception as e:
        print("Error is : ", e)
        return jsonify({"error": str(e)}), 500

@app.route('/callRoutes', methods=['GET'])
def callRoutes():
    try:
        # Get the language from the request (default to English)
        language = request.args.get('language', 'en')

        # Use the speech_recognition library to capture and convert speech to text
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            print("Listening...")
            audio = recognizer.listen(source)

        try:
            # Recognize speech (convert speech to text)
            if language == 'en':
                transcript = recognizer.recognize_google(audio, language='en-US')
            elif language == 'te':
                transcript = recognizer.recognize_google(audio, language='te-IN')
            elif language == 'hi':
                transcript = recognizer.recognize_google(audio, language='hi-IN')
            else:
                return jsonify({"error": "Unsupported language"}), 400

            # Translate the transcript to English (if not already in English)
            if language != 'en':
                try:
                    translation = translator.translate(transcript, src=language, dest='en')
                    transcript = translation.text if translation else 'Translation failed'
                except Exception as e:
                    print("Translation error:", e)
                    return jsonify({"error": "Translation failed"}), 500

            print("Transcript:", transcript)

            # Prompt for the generative model to respond based on the transcript
            prompt = f"I have these sections in my Sidebar, ['Profile', 'My Crops', 'Dashboard', 'Recommendation', 'Crop Disease', 'Market Price', 'Yield Prediction'].\nProfile - Shows the user profile\nMy Crops - The crops user has cultivated\nDashboard - User's Data and history\nRecommendation - Suggest which crop to cultivate based on the farm size\nCrop Disease - Detects which disease the crop has\nMarket Price - Suggests the market price of the crop user has\nYield prediction - predicts how much yield can be produced for user crops\n Based on the given user transcript '{transcript}', select one of the sections where the user should navigate and provide that as output. The output should be like 'Recommendation'. That's it, No other text should be present"

            # Use the generative model to generate content based on the prompt
            response = model.generate_content(prompt)
            generated_content = response.candidates[0].content.parts[0].text

            # Return the generated content as JSON
            return jsonify({"description": generated_content})

        except sr.UnknownValueError:
            return jsonify({"error": "Could not understand audio"}), 500
        except sr.RequestError as e:
            return jsonify({"error": f"Speech recognition error: {e}"}), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/voiceInputTelugu', methods=['GET'])
def voiceInputTelugu():
    try:
        # Use the speech_recognition library to capture and convert speech to text
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            print("Listening...")
            audio = recognizer.listen(source)

        try:
            # Recognize speech (convert speech to text)
            transcript = recognizer.recognize_google(audio, language='te-IN')

            # Translate the transcript to English
            try:
                translation = translator.translate(transcript, src='te', dest='en')
                transcript_en = translation.text if translation else 'Translation failed'
            except Exception as e:
                print("Translation error:", e)
                return jsonify({"error": "Translation failed"}), 500

            # Return the translated text as JSON
            return jsonify({"transcript": transcript_en})

        except sr.UnknownValueError:
            return jsonify({"error": "Could not understand audio"}), 500
        except sr.RequestError as e:
            return jsonify({"error": f"Speech recognition error: {e}"}), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/speakTelugu', methods=['POST'])
def speakTelugu():
    try:
        data = request.json
        text = data.get('text')
        if not text:
            return jsonify({"error": "No text provided"}), 400

        # Convert the text to speech in Telugu
        speak(text, 'te')

        return jsonify({"message": "Speech played successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True, port=9000)
