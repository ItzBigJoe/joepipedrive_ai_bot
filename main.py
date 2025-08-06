from flask import Flask, request, jsonify
import openai
import smtplib
from email.message import EmailMessage
import os
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)

@app.route("/webhook", methods=["POST"])
def webhook():
    data = request.json
    subject = data.get("subject")
    body = data.get("body")
    sender = data.get("from", {}).get("email_address")

    if not subject or not body or not sender:
        return jsonify({"error": "Missing required fields"}), 400

    try:
        # Generate reply
        prompt = f"Reply to this email: {body}"
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You're a helpful sales assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        reply = response['choices'][0]['message']['content']

        # Send to YOU for review
        send_review_email(subject, body, reply, sender)

        return jsonify({"status": "ok"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

def send_review_email(subject, original, reply, lead_email):
    msg = EmailMessage()
    msg["Subject"] = f"AI Draft Reply for: {subject}"
    msg["From"] = os.getenv("YOUR_EMAIL")
    msg["To"] = os.getenv("YOUR_EMAIL")
    msg["Reply-To"] = os.getenv("YOUR_EMAIL")
    msg.set_content(f"""
Original message from {lead_email}:
{original}

--- AI Suggested Reply ---
{reply}

Reply to this email if it's good.
""")

    with smtplib.SMTP_SSL(os.getenv("GMAIL_SMTP_SERVER"), 465) as smtp:
        smtp.login(os.getenv("YOUR_EMAIL"), os.getenv("EMAIL_APP_PASSWORD"))
        smtp.send_message(msg)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)

