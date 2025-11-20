from __future__ import print_function
import os.path
import base64
import re
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

# Gmail API scope
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

def get_gmail_service():
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    service = build('gmail', 'v1', credentials=creds)
    return service

def fetch_latest_emails(service, n=5):
    results = service.users().messages().list(userId='me', maxResults=n).execute()
    messages = results.get('messages', [])
    emails = []
    for msg in messages:
        txt = service.users().messages().get(userId='me', id=msg['id'], format='full').execute()
        headers = {h['name'].lower(): h['value'] for h in txt.get('payload', {}).get('headers', [])}
        subject = headers.get('subject', '')
        sender = headers.get('from', '')

        # try to extract a snippet/body
        snippet = txt.get('snippet', '')
        body = ''
        payload = txt.get('payload', {})
        parts = payload.get('parts', [])
        if parts:
            # find the text/plain part if present
            for p in parts:
                mime = p.get('mimeType', '')
                if mime == 'text/plain' and p.get('body', {}).get('data'):
                    body = base64.urlsafe_b64decode(p['body']['data']).decode('utf-8', errors='ignore')
                    break
            if not body:
                # fallback to first part's body
                try:
                    body = base64.urlsafe_b64decode(parts[0]['body'].get('data', '')).decode('utf-8', errors='ignore')
                except Exception:
                    body = ''
        else:
            body = base64.urlsafe_b64decode(payload.get('body', {}).get('data', '') or b'').decode('utf-8', errors='ignore') if payload.get('body', {}).get('data') else ''

        emails.append({
            'id': msg['id'],
            'subject': subject,
            'from': sender,
            'snippet': snippet,
            'body': body
        })
    return emails

if __name__ == '__main__':
    service = get_gmail_service()
    emails = fetch_latest_emails(service, n=3)
    for i, e in enumerate(emails, 1):
        print(f"\n--- Email {i} ---\n{e[:500]}")
