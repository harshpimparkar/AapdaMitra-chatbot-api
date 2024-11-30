import os
import logging
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from langdetect import detect
from groq import Groq
from groq._base_client import SyncHttpxClientWrapper
from flask_cors import CORS

load_dotenv()

app = Flask(__name__)

CORS(app, origins=["http://localhost:5173"])

http_client = SyncHttpxClientWrapper()

client = Groq(api_key=os.getenv("GROQ_API_KEY"), http_client=http_client)

NDRF_PROMPT = (
    "You are an NDRF officer addressing disaster queries. "
    "Respond in a professional, serious, and authoritative tone. "
    "Ensure that your responses prioritize public safety, providing clear and actionable information. "
    "You should respond in the same language in which you are addressed, such as Hindi, Tamil, Telugu, Kannada, Malayalam, Gujarati, Marathi, Bengali, Punjabi, or Kashmiri. "
    "Your goals are: "
    "1. Provide quick, accurate, and critical disaster information. "
    "2. Facilitate immediate access to emergency services. "
    "3. Promote preparedness and recovery using technology. "
    "Ensure responses are concise, clear, and keep the strict word limit of 200 words to ensure readability on mobile screens. "
    "Always include **numbered steps** when giving instructions or guidance, especially for actions that need to be taken during an emergency. "
    "The steps should be clear, actionable, and easy to follow, formatted in a numbered list. "
    "Keep the tone formal, professional, and focused on disaster management and public safety. "
    "If the question is not related to your domain, which is *Disasters in India and public safety of Indians*, then reply with: "
    "'I'm an NDRF officer, and my priority is to address disaster-related queries and provide critical information for public safety.' and stop right there."
)

DEFAULT_MODEL = "llama-3.1-70b-versatile"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 1024

logging.basicConfig(level=logging.DEBUG)

@app.route('/')
def home():
    app.logger.info('Accessed home route')
    return jsonify({"message": "Welcome to the NDRF Aapda Sahayta Bot!"})

@app.route('/v1/chat', methods=['POST'])
def generate_chat_response():
    try:
        data = request.get_json()
        messages = data.get('messages', [])

        if not messages or all(not msg.get("content") for msg in messages):
            return jsonify({
                "message": (
                    "Namaste, I'm NDRF Aapda Sahayta Bot. I'm here to help you with any queries or requests you may have during a disaster. "
                    "Please feel free to ask me anything in any of the following languages: Hindi, Tamil, Telugu, Kannada, Malayalam, Gujarati, Marathi, Bengali, Punjabi, or Kashmiri. \n\n"
                    "If you need assistance, please type 'help' and I will guide you through the process."
                ),
                "tokens_used": 0
            }), 200

        full_messages = [{"role": "system", "content": NDRF_PROMPT}]

        for msg in messages:
            msg["role"] = "user"
            user_language = detect(msg["content"])
            full_messages[0]["content"] = f"{NDRF_PROMPT} Respond in {user_language}."
            full_messages.append(msg)

        response = client.chat.completions.create(
            messages=full_messages,
            model=DEFAULT_MODEL,
            temperature=DEFAULT_TEMPERATURE,
            max_tokens=DEFAULT_MAX_TOKENS
        )

        assistant_message = response.choices[0].message.content

        assistant_message = format_response_for_mobile(assistant_message)

        return jsonify({
            "message": assistant_message,
            "tokens_used": response.usage.total_tokens
        }), 200

    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

def format_response_for_mobile(response):
    max_length = 300
    paragraphs = []

    while len(response) > max_length:
        split_point = response.rfind(' ', 0, max_length)
        paragraphs.append(response[:split_point])
        response = response[split_point:].strip()

    if response:
        paragraphs.append(response)

    return "\n\n".join(paragraphs)

@app.route('/v1/health', methods=['GET'])
def health_check():
    app.logger.info('Health check accessed')
    return jsonify({
        "status": "healthy",
        "message": "Aapda Sahayta Bot, COPY!"
    }), 200

if __name__ == '__main__':
    app.logger.info('Starting Flask app')
    app.run(debug=True, host='0.0.0.0', port=5000)