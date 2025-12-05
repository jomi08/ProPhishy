# backend/app.py
import os
import json
from typing import List, Dict
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from gmail_service import fetch_latest_messages, mark_as_read, delete_message
from hybrid_predict import classify_email   # using the new hybrid model
from pathlib import Path

app = FastAPI(title="ProPhishy backend")

# Allow frontend (adjust origin as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# simple persistence file to keep flagged items between restarts (optional)
DATA_PATH = Path(__file__).parent / "flagged.json"
_flagged: List[Dict] = []

def load_flagged():
    global _flagged
    if DATA_PATH.exists():
        try:
            _flagged = json.loads(DATA_PATH.read_text(encoding="utf-8")) or []
        except Exception:
            _flagged = []
    else:
        _flagged = []

def save_flagged():
    DATA_PATH.write_text(json.dumps(_flagged, ensure_ascii=False, indent=2), encoding="utf-8")

load_flagged()

@app.get("/metrics")
def get_metrics(max_results: int = 200):
    """
    Returns simple metrics computed from a recent fetch:
      - total: number of messages fetched (max_results)
      - safe: number of messages without 'UNREAD' label among fetched (treated as safe)
      - spam: number of flagged messages (persisted locally)
    """
    try:
        msgs = fetch_latest_messages(max_results=max_results)
    except Exception:
        # fallback to stored flagged data only
        return {"total": None, "safe": None, "spam": len(_flagged)}
    total = len(msgs)
    # count messages that do NOT have 'UNREAD' label => those are read
    safe = sum(1 for m in msgs if 'UNREAD' not in (m.get("labelIds") or []))
    spam = len(_flagged)
    return {"total": total, "safe": safe, "spam": spam}

@app.get("/flagged")
def get_flagged():
    # return the list of flagged emails
    return _flagged

@app.post("/refresh")
def refresh_emails(max_results: int = 25):
    """
    Fetch latest messages from Gmail and classify them. Returns the new flagged list.
    """
    try:
        messages = fetch_latest_messages(max_results=max_results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    changed = False

    for msg in messages:
        subject = msg.get("subject", "") or ""
        body = msg.get("body") or msg.get("snippet", "") or ""

        # HYBRID MODEL: subject TF-IDF + body BiLSTM
        pred = classify_email(subject, body)

        # DEBUG PRINT
        print("\n------------------------------------")
        print("EMAIL:", subject[:100])
        print("  p_subject:", pred.get("p_subject"))
        print("  p_body:", pred.get("p_body"))
        print("  final_score:", pred.get("score"))
        print("  label:", pred.get("label"))
        print("------------------------------------")

        entry = {
            "id": msg["id"],
            "threadId": msg.get("threadId"),
            "from": msg.get("from"),
            "subject": msg.get("subject"),
            "snippet": msg.get("snippet"),
            "body": msg.get("body"),
            "score": pred.get("score"),
            "label": pred.get("label"),
            "p_subject": pred.get("p_subject"),
            "p_body": pred.get("p_body"),
            "raw": None
        }

        exists = next((x for x in _flagged if x["id"] == entry["id"]), None)

        if exists:
            exists.update(entry)
            changed = True
        else:
            if pred.get("label") == "spam":
                _flagged.insert(0, entry)
                changed = True

    if changed:
        save_flagged()

    return {"flagged_count": len(_flagged), "flagged": _flagged}


@app.post("/action/mark-safe")
def mark_safe(payload: Dict):
    """
    Remove an email from flagged (user marks it safe).
    payload: {"id": "<message id>"}
    """
    mid = payload.get("id")
    if not mid:
        raise HTTPException(status_code=400, detail="missing id")
    global _flagged
    before = len(_flagged)
    _flagged = [f for f in _flagged if f["id"] != mid]
    if len(_flagged) != before:
        save_flagged()
    return {"ok": True, "flagged_count": len(_flagged)}

@app.post("/action/delete")
def action_delete(payload: Dict):
    """
    Delete (trash) message in Gmail and remove from flagged list.
    payload: {"id": "<message id>"}
    """
    mid = payload.get("id")
    if not mid:
        raise HTTPException(status_code=400, detail="missing id")
    try:
        delete_message(mid)
    except Exception as e:
        # still proceed to remove locally
        print("delete_message error:", e)
    global _flagged
    _flagged = [f for f in _flagged if f["id"] != mid]
    save_flagged()
    return {"ok": True, "flagged_count": len(_flagged)}
