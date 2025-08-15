from flask import Flask, request, jsonify
from openai import OpenAI
import os
from dotenv import load_dotenv
import faiss
import numpy as np
import pickle
from difflib import unified_diff
from openai.types.chat import ChatCompletion
from db import init_db, save_reply, get_recent_replies


# Load environment variables
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

FAISS_INDEX_FILE = "vector_index.faiss"
METADATA_FILE = "vector_metadata.pkl"
DRAFTS_FILE = "pending_drafts.pkl"

# Preferred models in order of capability
preferred_models = ["gpt-4o", "gpt-4", "gpt-4o-mini", "gpt-3.5-turbo"]

def get_first_available_model():
    try:
        available_models = [m.id for m in client.models.list().data]
        for model in preferred_models:
            if model in available_models:
                return model
        raise Exception("No preferred models available to this API key.")
    except Exception as e:
        raise Exception(f"Error checking available models: {str(e)}")

try:
    model_to_use = get_first_available_model()
    print(f"✅ Using model: {model_to_use}")
except Exception as e:
    print(f"❌ {e}")
    exit(1)

# ================================
# FAISS INIT
# ================================
if os.path.exists(FAISS_INDEX_FILE):
    index = faiss.read_index(FAISS_INDEX_FILE)
    with open(METADATA_FILE, "rb") as f:
        metadata = pickle.load(f)
else:
    index = faiss.IndexFlatL2(1536)
    metadata = []

# Load pending drafts
if os.path.exists(DRAFTS_FILE):
    with open(DRAFTS_FILE, "rb") as f:
        pending_drafts = pickle.load(f)
else:
    pending_drafts = {}  # key: email_id, value: draft data

def save_all():
    faiss.write_index(index, FAISS_INDEX_FILE)
    with open(METADATA_FILE, "wb") as f:
        pickle.dump(metadata, f)
    with open(DRAFTS_FILE, "wb") as f:
        pickle.dump(pending_drafts, f)

# ================================
# EMBEDDINGS
# ================================
def get_embedding(text):
    embedding = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    ).data[0].embedding
    return np.array(embedding, dtype=np.float32)

def add_to_vector_db(subject, body, your_reply, ai_draft=None):
    """Store the final reply and optionally the AI's original draft for learning."""
    text_to_store = f"Email:\n{body}\nFinal Reply:\n{your_reply}"
    if ai_draft:
        text_to_store += f"\nAI Draft:\n{ai_draft}"
    vector = get_embedding(body)
    index.add(np.array([vector]))
    metadata.append({
        "subject": subject,
        "body": body,
        "reply": your_reply,
        "ai_draft": ai_draft
    })
    save_all()

def search_similar(body, k=3):
    if len(metadata) == 0:
        return []
    vector = get_embedding(body)
    distances, ids = index.search(np.array([vector]), k)
    results = [metadata[i] for i in ids[0] if i < len(metadata)]
    return results

# ================================
# FLASK APP
# ================================
app = Flask(__name__)

@app.route("/webhook", methods=["POST"])
def webhook():
    data = request.json
    subject = data.get("subject", "")
    body = data.get("body", "")
    email_id = data.get("email_id", str(len(pending_drafts) + 1))  # unique ID

    # Get examples from similar replies
    examples = search_similar(body, k=3)
    examples_text = "\n\n".join(
        [f"Email:\n{ex['body']}\nYour reply:\n{ex['reply']}" for ex in examples]
    )

    # Also include examples of AI Draft → Human Edit differences
    edit_examples = [
        f"Original AI Draft:\n{ex['ai_draft']}\nHuman Edit:\n{ex['reply']}"
        for ex in examples if ex.get("ai_draft")
    ]
    edit_text = "\n\n".join(edit_examples)

    try:
        prompt = (
            "You are a helpful sales assistant. "
            "Here are past examples of my style:\n\n"
            f"{examples_text}\n\n"
            "Here are examples of how I typically edit AI drafts:\n\n"
            f"{edit_text}\n\n"
            f"Now, reply to this new email:\n{body}"
        )

        chat_response = client.chat.completions.create(
            model=model_to_use,
            messages=[
                {"role": "system", "content": "You're a helpful sales assistant who mimics my style."},
                {"role": "user", "content": prompt}
            ]
        )
        ai_reply = chat_response.choices[0].message.content

        # Save AI draft temporarily
        pending_drafts[email_id] = {
            "subject": subject,
            "body": body,
            "ai_draft": ai_reply
        }
        save_all()

        return jsonify({"reply": ai_reply, "email_id": email_id}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/save-reply", methods=["POST"])
def save_reply_route():
    data = request.json
    email_id = data.get("email_id")
    your_reply = data.get("your_reply", "")

    if email_id not in pending_drafts:
        return jsonify({"error": "No draft found for this email_id"}), 404

    draft_data = pending_drafts.pop(email_id)
    subject = draft_data["subject"]
    body = draft_data["body"]
    ai_draft = draft_data["ai_draft"]

    # Store in vector DB with AI draft for future learning
    add_to_vector_db(subject, body, your_reply, ai_draft)

    # Show difference for debugging (optional)
    diff = "\n".join(unified_diff(
        ai_draft.splitlines(),
        your_reply.splitlines(),
        fromfile="AI Draft",
        tofile="Your Final Reply",
        lineterm=""
    ))
    print(f"✏️ Edit Difference for {email_id}:\n{diff}")

    return jsonify({"status": "saved", "diff": diff}), 200

if __name__ == "__main__":
    init_db()
    app.run(debug=True)