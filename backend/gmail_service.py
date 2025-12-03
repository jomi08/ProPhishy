# gmail_service.py
import os
import base64
import json
import time
from typing import List, Dict, Optional
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from email import policy
from email.parser import BytesParser

# Scopes: readonly for reading mail, modify for marking read/delete if needed
SCOPES = ["https://www.googleapis.com/auth/gmail.readonly",
          "https://www.googleapis.com/auth/gmail.modify"]

CREDENTIALS_PATH = os.path.join(os.path.dirname(__file__), "credentials.json")
TOKEN_PATH = os.path.join(os.path.dirname(__file__), "token.json")


def ensure_credentials() -> Credentials:
    creds = None
    if os.path.exists(TOKEN_PATH):
        creds = Credentials.from_authorized_user_file(TOKEN_PATH, SCOPES)
    # If there are no valid credentials, do the OAuth flow
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists(CREDENTIALS_PATH):
                raise FileNotFoundError("credentials.json not found in backend/ directory")
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_PATH, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(TOKEN_PATH, "w") as f:
            f.write(creds.to_json())
    return creds


def get_service():
    creds = ensure_credentials()
    service = build("gmail", "v1", credentials=creds, cache_discovery=False)
    return service


def extract_plain_text_from_message(message_payload) -> str:
    """
    Given a raw message payload (from Gmail API 'get' with format='raw' or 'full'),
    attempt to extract a readable plain text string.
    """
    # Try to handle raw first
    if "raw" in message_payload:
        try:
            raw = base64.urlsafe_b64decode(message_payload["raw"])
            msg = BytesParser(policy=policy.default).parsebytes(raw)
            # prefer text/plain
            if msg.get_body(preferencelist=("plain",)):
                return msg.get_body(preferencelist=("plain",)).get_content()
            # fallback to html body
            if msg.get_body(preferencelist=("html",)):
                return msg.get_body(preferencelist=("html",)).get_content()
            payload = msg.get_payload(decode=True)
            if isinstance(payload, bytes):
                return payload.decode(errors="ignore")
            return str(payload)
        except Exception:
            # fall through to payload-based extraction
            pass

    # If message_payload is a 'full' representation with parts
    payload = message_payload.get("payload", {})

    def walk_parts(part):
        mime = part.get("mimeType", "")
        if mime == "text/plain" and "body" in part and part["body"].get("data"):
            raw = part["body"]["data"]
            try:
                return base64.urlsafe_b64decode(raw).decode(errors="ignore")
            except Exception:
                return base64.b64decode(raw).decode(errors="ignore")
        # if multipart, go deeper
        for p in part.get("parts", []) or []:
            text = walk_parts(p)
            if text:
                return text
        return None

    text = walk_parts(payload)
    if text:
        return text
    # fallback: try snippet
    return message_payload.get("snippet", "") or ""


def fetch_latest_messages(max_results: int = 25, label_ids: Optional[List[str]] = None) -> List[Dict]:
    """
    Fetch the latest messages (list -> get each)
    Returns a list of message dicts with keys:
      id, threadId, from, subject, snippet, body, internalDate, raw, labelIds, read
    NOTE: this now ensures labelIds are returned (via format='full') and provides a boolean 'read'.
    """
    service = get_service()
    query_label = label_ids if label_ids else ["INBOX"]
    # Use list to get ids; list may not include labelIds for each message, so we do a get() per id.
    results = service.users().messages().list(userId="me", maxResults=max_results, labelIds=query_label).execute()
    messages = results.get("messages", []) or []

    out = []
    for m in messages:
        mid = m["id"]
        try:
            # get message in full format so labelIds and payload are present
            msg = service.users().messages().get(userId="me", id=mid, format="full").execute()
        except Exception:
            # fallback: if get fails, skip this message
            continue

        headers = {h["name"].lower(): h["value"] for h in msg.get("payload", {}).get("headers", [])}
        frm = headers.get("from", "")
        subject = headers.get("subject", "")
        snippet = msg.get("snippet", "")
        raw_text = extract_plain_text_from_message(msg)
        internalDate = int(msg.get("internalDate", 0))
        label_ids_msg = msg.get("labelIds", []) or []

        # Determine read status: treat as read if labelIds exist and UNREAD not present
        is_read = False
        if label_ids_msg:
            is_read = "UNREAD" not in label_ids_msg
        else:
            # If no labelIds are present for some reason, we do not assume unread; mark as False
            # (higher-level code can decide fallback behavior)
            is_read = False

        out.append({
    "id": mid,
    "threadId": msg.get("threadId"),
    "from": frm,
    "subject": subject,
    "snippet": snippet,
    "body": raw_text,
    "internalDate": internalDate,
    "labelIds": msg.get("labelIds", []),
    "raw": msg
})

    return out


def mark_as_read(message_id: str):
    service = get_service()
    service.users().messages().modify(userId="me", id=message_id, body={"removeLabelIds": ["UNREAD"]}).execute()


def delete_message(message_id: str):
    service = get_service()
    service.users().messages().trash(userId="me", id=message_id).execute()
