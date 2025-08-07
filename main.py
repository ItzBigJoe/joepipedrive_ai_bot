from flask import Flask, request, jsonify
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)


app = Flask(__name__)

@app.route("/webhook", methods=["POST"])
def webhook():
    data = request.json
    subject = data.get("subject")
    body = data.get("body")

    try:
        prompt = f"Reply to this email: {body}"
        chat_response = client.chat.completions.create(
            model="gpt-4o",
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