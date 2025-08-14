from flask import Flask, request, jsonify
from openai import OpenAI
import os
from dotenv import load_dotenv
from openai.types.chat import ChatCompletion

# Load environment variables
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Preferred models in order of capability
preferred_models = ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"]

def get_first_available_model():
    """Check which preferred model is available for this API key."""
    try:
        available_models = [m.id for m in client.models.list().data]
        for model in preferred_models:
            if model in available_models:
                return model
        raise Exception("No preferred models available to this API key.")
    except Exception as e:
        raise Exception(f"Error checking available models: {str(e)}")

# Pre-select model at startup
try:
    model_to_use = get_first_available_model()
    print(f"✅ Using model: {model_to_use}")
except Exception as e:
    print(f"❌ {e}")
    exit(1)

app = Flask(__name__)

@app.route("/webhook", methods=["POST"])
def webhook():
    data = request.json
    subject = data.get("subject", "")
    body = data.get("body", "")

    try:
        prompt = f"Reply to this email: {body}"
        chat_response: ChatCompletion = client.chat.completions.create(
            model=model_to_use,
            messages=[
                {"role": "system", "content": "You're a helpful sales assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        reply = chat_response.choices[0].message.content
        return jsonify({"reply": reply}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
