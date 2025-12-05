# backend/app.py
import os
import json
from typing import List, Dict
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from gmail_service import fetch_latest_messages, mark_as_read, delete_message
from hybrid_predict_v2 import classify_email   # using the new hybrid model with BiLSTM + Attention
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

# simple persistence files
DATA_PATH = Path(__file__).parent / "flagged.json"
ALL_EMAILS_PATH = Path(__file__).parent / "all_emails.json"
_flagged: List[Dict] = []
_all_emails: List[Dict] = []

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

def load_all_emails():
    global _all_emails
    if ALL_EMAILS_PATH.exists():
        try:
            _all_emails = json.loads(ALL_EMAILS_PATH.read_text(encoding="utf-8")) or []
        except Exception:
            _all_emails = []
    else:
        _all_emails = []

def save_all_emails():
    ALL_EMAILS_PATH.write_text(json.dumps(_all_emails, ensure_ascii=False, indent=2), encoding="utf-8")

load_flagged()
load_all_emails()

@app.get("/metrics")
def get_metrics():
    """
    Returns metrics computed from stored emails:
      - total: total number of emails in storage
      - safe: number of emails marked as safe (label != 'spam' OR marked_safe == True)
      - spam: number of emails marked as spam (label == 'spam' AND marked_safe == False)
    """
    total = len(_all_emails)
    spam = sum(1 for e in _all_emails if e.get("label") == "spam" and not e.get("marked_safe", False))
    safe = total - spam
    return {"total": total, "safe": safe, "spam": spam}

@app.get("/flagged")
def get_flagged():
    # return the list of flagged emails
    return _flagged

@app.get("/emails")
def get_emails(max_results: int = 50):
    """
    Returns all emails (stored or freshly fetched from both INBOX and SPAM).
    If stored emails exist, return those. Otherwise fetch fresh.
    """
    if _all_emails:
        return _all_emails
    
    # Fetch fresh if no stored emails
    try:
        print(f"No stored emails, fetching {max_results} from INBOX...")
        inbox_msgs = fetch_latest_messages(max_results=max_results, label_ids=["INBOX"])
        print(f"Fetching {max_results} from SPAM...")
        spam_msgs = fetch_latest_messages(max_results=max_results, label_ids=["SPAM"])
        
        # Combine and deduplicate
        messages = inbox_msgs + spam_msgs
        seen_ids = set()
        unique_messages = []
        for msg in messages:
            if msg["id"] not in seen_ids:
                seen_ids.add(msg["id"])
                unique_messages.append(msg)
        
        # Classify each email
        result = []
        for msg in unique_messages:
            subject = msg.get("subject", "") or ""
            body = msg.get("body") or msg.get("snippet", "") or ""
            pred = classify_email(subject, body)
            
            entry = {
                "id": msg["id"],
                "from": msg.get("from"),
                "subject": msg.get("subject"),
                "body": msg.get("body"),
                "score": pred.get("score"),
                "label": pred.get("label"),
                "marked_safe": False
            }
            result.append(entry)
        
        return result
    except Exception as e:
        print(f"Error fetching emails: {e}")
        return []

@app.post("/refresh")
def refresh_emails(max_results: int = 25):
    """
    Fetch latest messages from both INBOX and SPAM, classify them, and return metrics.
    """
    try:
        # Fetch from both INBOX and SPAM
        print(f"Fetching {max_results} messages from INBOX...")
        inbox_msgs = fetch_latest_messages(max_results=max_results, label_ids=["INBOX"])
        print(f"Fetching {max_results} messages from SPAM...")
        spam_msgs = fetch_latest_messages(max_results=max_results, label_ids=["SPAM"])
        
        # Combine and deduplicate by ID
        messages = inbox_msgs + spam_msgs
        seen_ids = set()
        unique_messages = []
        for msg in messages:
            if msg["id"] not in seen_ids:
                seen_ids.add(msg["id"])
                unique_messages.append(msg)
        messages = unique_messages
        
        print(f"Total unique messages after combining INBOX + SPAM: {len(messages)}")
    except Exception as e:
        print(f"Error fetching messages: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    changed = False
    spam_count = 0
    safe_count = 0

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
            "marked_safe": False,
            "raw": None
        }

        # Update or add to all_emails list
        exists_in_all = next((x for x in _all_emails if x["id"] == entry["id"]), None)
        if exists_in_all:
            exists_in_all.update(entry)
        else:
            _all_emails.insert(0, entry)
            changed = True

        # Update flagged list (only spam)
        exists_in_flagged = next((x for x in _flagged if x["id"] == entry["id"]), None)
        if pred.get("label") == "spam":
            spam_count += 1
            if exists_in_flagged:
                exists_in_flagged.update(entry)
            else:
                _flagged.insert(0, entry)
        else:
            safe_count += 1
            # Remove from flagged if it was marked safe
            if exists_in_flagged:
                _flagged.remove(exists_in_flagged)
                changed = True

    if changed:
        save_flagged()
        save_all_emails()

    # Return metrics
    return {
        "total": len(_all_emails),
        "spam": spam_count,
        "safe": safe_count,
        "flagged_count": len(_flagged)
    }


@app.post("/action/mark-safe")
def mark_safe(payload: Dict):
    """
    Mark an email as safe (user overrides spam classification).
    payload: {"id": "<message id>"}
    """
    mid = payload.get("id")
    if not mid:
        raise HTTPException(status_code=400, detail="missing id")
    
    global _flagged, _all_emails
    
    # Update marked_safe flag in all_emails
    email_entry = next((e for e in _all_emails if e["id"] == mid), None)
    if email_entry:
        email_entry["marked_safe"] = True
        save_all_emails()
    
    # Remove from flagged list
    before = len(_flagged)
    _flagged = [f for f in _flagged if f["id"] != mid]
    if len(_flagged) != before:
        save_flagged()
    
    # Recalculate metrics
    spam_count = sum(1 for e in _all_emails if e.get("label") == "spam" and not e.get("marked_safe", False))
    
    return {
        "ok": True, 
        "flagged_count": len(_flagged),
        "spam_count": spam_count,
        "total": len(_all_emails)
    }

@app.post("/action/delete")
def action_delete(payload: Dict):
    """
    Delete (trash) message in Gmail and remove from all lists.
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
    
    global _flagged, _all_emails
    _flagged = [f for f in _flagged if f["id"] != mid]
    _all_emails = [e for e in _all_emails if e["id"] != mid]
    save_flagged()
    save_all_emails()
    
    return {"ok": True, "flagged_count": len(_flagged), "total": len(_all_emails)}
