from fastapi import FastAPI
from backend.gmail_service import get_gmail_service, fetch_latest_emails
from backend.predict import predict_email

app = FastAPI()
@app.get("/")
def root():
    return {"message": "🚀 FastAPI is running! Go to /docs to test the endpoints."}


@app.get("/check-emails")
def check_emails():
    service = get_gmail_service()
    service = get_gmail_service()
    emails = fetch_latest_emails(service, n=10)
    results = []
    for e in emails:
        # e is now a dict with subject, from, snippet, body
        text = e.get('body') or e.get('snippet') or ''
        pred = predict_email(text)
        results.append({
            'id': e.get('id'),
            'subject': e.get('subject'),
            'from': e.get('from'),
            'snippet': e.get('snippet'),
            'score': pred.get('score'),
            'label': pred.get('label')
        })
    return results


@app.get('/metrics')
def metrics():
    # simple aggregates for demo purposes
    service = get_gmail_service()
    emails = fetch_latest_emails(service, n=50)
    flagged = 0
    today_flagged = 0
    for e in emails:
        text = e.get('body') or e.get('snippet') or ''
        pred = predict_email(text)
        if pred.get('label') == 'Phishing':
            flagged += 1
    return {
        'flagged_count': flagged,
        'total_checked': len(emails),
        'today_flagged': flagged  # placeholder — backend can compute per-day
    }


@app.get('/flagged')
def flagged():
    service = get_gmail_service()
    emails = fetch_latest_emails(service, n=50)
    results = []
    for e in emails:
        text = e.get('body') or e.get('snippet') or ''
        pred = predict_email(text)
        if pred.get('label') == 'Phishing':
            results.append({
                'id': e.get('id'),
                'subject': e.get('subject'),
                'from': e.get('from'),
                'snippet': e.get('snippet'),
                'body': e.get('body'),
                'score': pred.get('score')
            })
    return results


@app.post('/refresh')
def refresh():
    # For the demo this simply returns ok. Integrate with background job if needed.
    return {'ok': True}


if __name__ == '__main__':
    import uvicorn
    # Run on port 5000 by default to match frontend proxy expectation
    uvicorn.run("app:app", host="0.0.0.0", port=5000, log_level="info")
