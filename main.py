# main.py

from flask import Flask, request, jsonify
from bs4 import BeautifulSoup
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
FAISS_INDEX_FILE = "index.faiss"
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

#FAISS_INDEX_FILE = "index.faiss"

def save_all():
    print(">>> DEBUG save_all")
    print("Type of index:", type(index))
    print("FAISS_INDEX_FILE:", FAISS_INDEX_FILE, type(FAISS_INDEX_FILE))
    faiss.write_index(index, str(FAISS_INDEX_FILE))  # force string path
    _atomic_pickle_dump(metadata, METADATA_FILE)
    _atomic_pickle_dump(pending_drafts, DRAFTS_FILE)

print("FAISS_INDEX_FILE:", FAISS_INDEX_FILE, type(FAISS_INDEX_FILE))

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

import traceback

@app.route("/webhook", methods=["POST"])
def webhook():
    try:
        if not request.is_json:
            return jsonify({"error": "Expected application/json body"}), 400

        data = request.get_json(silent=True) or {}
        subject = data.get("subject", "").strip()

        raw_html = data.get("body", "").strip()
        plain_text = data.get("body_plain", "").strip() if data.get("body_plain") else None

        if plain_text:
            body = plain_text
        else:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(raw_html, "html.parser")
            body = soup.get_text(separator="\n").strip()

        email_id = str(data.get("email_id", "") or (len(pending_drafts) + 1))

        if not body:
            return jsonify({"error": "Missing required field: body"}), 400

        # --- Your AI generation logic here (same as before) ---
        ai_reply = f"[MOCK REPLY] Test reply for: {body[:50]}..."

        pending_drafts[email_id] = {
            "subject": subject,
            "body_html": raw_html,
            "body_clean": body,
            "ai_draft": ai_reply
        }

        save_all()
        print(f"📝 Draft generated for email_id={email_id}")

        return jsonify({"reply": ai_reply, "email_id": email_id}), 200

    except Exception as e:
        error_message = f"Webhook failed: {str(e)}"
        print(error_message)
        print(traceback.format_exc())  # full stack trace
        return jsonify({"error": error_message}), 500


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
    
@app.route("/")
def index():
    return app.send_static_file("index.html")  # serve from templates
    
@app.route("/pending-drafts", methods=["GET"])
def get_pending_drafts():
    return jsonify(pending_drafts)



if __name__ == "__main__":
    init_db()
    # Avoid reloader double-running with FAISS/pickles.
    app.run(debug=True, use_reloader=False)
