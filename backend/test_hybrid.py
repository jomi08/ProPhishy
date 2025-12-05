from hybrid_predict import classify_email

tests = [
    ("Security alert", "Your account has been locked. Click here to verify."),
    ("Meeting tomorrow", "Hi team, let's meet tomorrow at 10 am in the office."),
    ("You won a prize", "Congratulations! Click the link to claim your gift card now."),
]

for subj, body in tests:
    print(subj, "->", classify_email(subj, body))
