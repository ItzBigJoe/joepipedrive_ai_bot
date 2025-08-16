# main.py

from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
import os
from dotenv import load_dotenv
import faiss
import numpy as np
import pickle
from difflib import unified_diff
from db import init_db, save_reply as save_reply_sql

# -----------------------------
# Load environment & OpenAI
# -----------------------------
load_dotenv()

MOCK_MODE = True  # <-- set to False to use real API
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set. Please add it to your environment or .env file.")

client = OpenAI(api_key=OPENAI_API_KEY)

# -----------------------------
# Files & constants
# -----------------------------
FAISS_INDEX_FILE = "vector_index.faiss"
METADATA_FILE = "vector_metadata.pkl"
DRAFTS_FILE = "pending_drafts.pkl"
EMBED_DIM = 1536  # text-embedding-3-small = 1536 dims

# Preferred models in order of capability
preferred_models = ["gpt-4o", "gpt-4", "gpt-4o-mini", "gpt-3.5-turbo"]
DEFAULT_FALLBACK_MODEL = "gpt-4o-mini"

def get_first_available_model():
    """
    Try to find the first preferred model available to this key.
    If listing is blocked on the plan/key, fall back safely.
    """
    try:
        # Some plans/keys can’t list models; handle gracefully.
        model_ids = {m.id for m in client.models.list().data}
        for m in preferred_models:
            if m in model_ids:
                return m
        # If we can list but none are in the list, last resort:
        return DEFAULT_FALLBACK_MODEL
    except Exception:
        # If listing is blocked, fall back to a reasonable default.
        return DEFAULT_FALLBACK_MODEL

model_to_use = get_first_available_model()
print(f"✅ Using model: {model_to_use}")

# -----------------------------
# FAISS init + metadata
# -----------------------------
def _safe_load_pickle(path, default):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return default

if os.path.exists(FAISS_INDEX_FILE):
    try:
        index = faiss.read_index(FAISS_INDEX_FILE)
        if index.d != EMBED_DIM:
            print("⚠️ FAISS index dim mismatch; reinitializing.")
            index = faiss.IndexFlatL2(EMBED_DIM)
    except Exception as e:
        print(f"⚠️ Failed to read FAISS index ({e}); reinitializing.")
        index = faiss.IndexFlatL2(EMBED_DIM)
else:
    index = faiss.IndexFlatL2(EMBED_DIM)

metadata = _safe_load_pickle(METADATA_FILE, [])
if not isinstance(metadata, list):
    print("⚠️ Metadata corrupted; resetting.")
    metadata = []

# Keep lengths in sync
if index.ntotal != len(metadata):
    print(f"⚠️ Index/metadata length mismatch ({index.ntotal} vs {len(metadata)}). Truncating to min.")
    min_len = min(index.ntotal, len(metadata))
    if index.ntotal > min_len:
        # Rebuild a smaller index from scratch (we don't have vectors; drop extra from index)
        # Easiest: rebuild from embeddings if you stored them. Since we didn't, trim metadata.
        metadata = metadata[:min_len]
    elif len(metadata) > min_len:
        metadata = metadata[:min_len]

pending_drafts = _safe_load_pickle(DRAFTS_FILE, {})
if not isinstance(pending_drafts, dict):
    print("⚠️ Drafts store corrupted; resetting.")
    pending_drafts = {}

def _atomic_pickle_dump(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(obj, f)
    os.replace(tmp, path)

def save_all():
    faiss.write_index(index, FAISS_INDEX_FILE)
    _atomic_pickle_dump(metadata, METADATA_FILE)
    _atomic_pickle_dump(pending_drafts, DRAFTS_FILE)

# -----------------------------
# Embeddings
# -----------------------------

def get_embedding(text: str) -> np.ndarray:
    if MOCK_MODE:
        # Just return a random vector of correct dimension
        return np.random.rand(EMBED_DIM).astype(np.float32)
    text = (text or "").strip()
    try:
        emb = client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        ).data[0].embedding
        return np.asarray(emb, dtype=np.float32)
    except Exception as e:
        raise RuntimeError(f"Embedding failed: {e}")

#def get_embedding(text: str) -> np.ndarray:
   # text = (text or "").strip()
    #try:
       # emb = client.embeddings.create(
            #model="text-embedding-3-small",
           # input=text
       # ).data[0].embedding
     #   return np.asarray(emb, dtype=np.float32)
    #except Exception as e:
  #      raise RuntimeError(f"Embedding failed: {e}")

def add_to_vector_db(subject: str, body: str, your_reply: str, ai_draft: str | None = None):
    """Store the final reply + (optionally) the AI's original draft for future learning."""
    vec = get_embedding(body or "")
    index.add(np.array([vec]))
    metadata.append({
        "subject": subject or "",
        "body": body or "",
        "reply": your_reply or "",
        "ai_draft": ai_draft or ""
    })
    save_all()

def search_similar(body: str, k: int = 3):
    if index.ntotal == 0 or len(metadata) == 0:
        return []
    k = max(1, min(k, index.ntotal))
    query_vec = get_embedding(body or "")
    distances, ids = index.search(np.array([query_vec]), k)
    id_list = [i for i in ids[0] if 0 <= i < len(metadata)]
    return [metadata[i] for i in id_list]

# -----------------------------
# Flask app
# -----------------------------
app = Flask(__name__)
CORS(app)  # Allow Zapier/Pipedrive to call your endpoints

@app.route("/webhook", methods=["POST"])
def webhook():
    try:
        # Try JSON first
        if request.is_json:
            data = request.get_json(silent=True) or {}
        else:
            # Fallback: handle form-encoded or other
            data = request.form.to_dict() or {}
            if not data:
                data = {"raw_body": request.get_data(as_text=True)}

        subject = data.get("subject", "").strip()
        body = data.get("body", "").strip()
        email_id = str(data.get("email_id", "") or (len(pending_drafts) + 1))

        if not body:
            return jsonify({
                "error": "Missing required field: body",
                "received_data": data
            }), 400

        # Retrieve style examples
        examples = search_similar(body, k=3)
        examples_text = "\n\n".join(
            [f"Email:\n{ex.get('body')}\nYour reply:\n{ex.get('reply')}" 
             for ex in examples if ex.get("body") and ex.get("reply")]
        )

        edit_text = "\n\n".join(
            [f"Original AI Draft:\n{ex.get('ai_draft')}\nHuman Edit:\n{ex.get('reply')}" 
             for ex in examples if ex.get("ai_draft") and ex.get("reply")]
        )

        prompt_parts = [
            "You are a helpful sales assistant. Mirror my tone and style.",
        ]
        if examples_text:
            prompt_parts.append("Here are past examples:\n\n" + examples_text)
        if edit_text:
            prompt_parts.append("Here are examples of edits:\n\n" + edit_text)
        prompt_parts.append("Now, reply to this email:\n" + body)

        prompt = "\n\n".join(prompt_parts)

        if MOCK_MODE:
            ai_reply = f"[MOCK REPLY] Fake AI reply for: {body[:50]}..."
        else:
            chat_response = client.chat.completions.create(
                model=model_to_use,
                messages=[
                    {"role": "system", "content": "You're a helpful sales assistant who mimics my style."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.4
            )
            ai_reply = (chat_response.choices[0].message.content or "").strip()

        pending_drafts[email_id] = {
            "subject": subject,
            "body": body,
            "ai_draft": ai_reply
        }
        save_all()
        print(f"📝 Draft generated for email_id={email_id}")

        return jsonify({
            "status": "ok",
            "reply": ai_reply,
            "email_id": email_id,
            "received_data": data  # helps debugging in Zapier
        }), 200

    except Exception as e:
        return jsonify({"error": f"Webhook processing failed: {e}"}), 500


@app.route("/save-reply", methods=["POST"])
def save_reply_route():
    if not request.is_json:
        return jsonify({"error": "Expected application/json body"}), 400

    data = request.get_json(silent=True) or {}
    email_id = str(data.get("email_id", "")).strip()
    your_reply = (data.get("your_reply", "") or "").strip()

    if not email_id:
        return jsonify({"error": "Missing required field: email_id"}), 400
    if not your_reply:
        return jsonify({"error": "Missing required field: your_reply"}), 400

    if email_id not in pending_drafts:
        return jsonify({"error": f"No draft found for email_id={email_id}"}), 404

    draft_data = pending_drafts.pop(email_id)
    subject = draft_data.get("subject", "")
    body = draft_data.get("body", "")
    ai_draft = (draft_data.get("ai_draft") or "").strip()

    # Write to vector DB (for learning) + SQLite (for record keeping)
    add_to_vector_db(subject, body, your_reply, ai_draft)
    try:
        save_reply_sql(subject, body, your_reply)
    except Exception as e:
        # Non-fatal; vector DB is already updated
        print(f"⚠️ SQLite save failed: {e}")

    # Show difference for debugging (optional)
    diff = "\n".join(unified_diff(
        (ai_draft or "").splitlines(),
        (your_reply or "").splitlines(),
        fromfile="AI Draft",
        tofile="Your Final Reply",
        lineterm=""
    ))
    print(f"✏️ Edit Difference for {email_id}:\n{diff}")

    save_all()
    return jsonify({"status": "saved", "diff": diff}), 200

if __name__ == "__main__":
    init_db()
    # Avoid reloader double-running with FAISS/pickles.
    app.run(debug=True, use_reloader=False)
