import os

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from google import genai

load_dotenv()

app = Flask(__name__)

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")

if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY is not configured in the .env file.")

client = genai.Client(api_key=API_KEY)

config_path = os.path.join(os.path.dirname(__file__), "chatbot_config.txt")
with open(config_path, "r", encoding="utf-8") as file:
    SYSTEM_PROMPT = file.read().strip()


@app.route("/")
def index():
    return render_template("index.html")


@app.post("/chat")
def chat():
    data = request.get_json(silent=True) or {}
    user_message = str(data.get("message", "")).strip()

    if not user_message:
        return jsonify({"error": "Please enter a question."}), 400

    try:
        prompt = f"{SYSTEM_PROMPT}\n\nUSER QUESTION:\n{user_message}"

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
        )

        answer = getattr(response, "text", None)

        if not answer:
            return jsonify({"error": "The model returned an empty response."}), 502

        return jsonify({"answer": answer.strip()})

    except Exception:
        app.logger.exception("Gemini API request failed")
        return jsonify({
            "error": "Unable to process the request right now. Please try again."
        }), 500


if __name__ == "__main__":
    app.run(debug=False)
