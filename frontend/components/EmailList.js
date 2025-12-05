import Card from './Card'
import EmailModal from './EmailModal'

export default function EmailList({ emails }) {

  // -----------------------------
  // Mark Safe Function (via Next proxy)
  // -----------------------------
  const markSafe = async (id) => {
    try {
      await fetch('/api/proxy/action/mark-safe', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ id })
      })
      alert('Email marked as safe!')
    } catch (err) {
      console.error('markSafe error:', err)
    }
  }

  // -----------------------------
  // Delete Email Function (via Next proxy)
  // -----------------------------
  const deleteEmail = async (id) => {
    try {
      await fetch('/api/proxy/action/delete', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ id })
      })
      alert('Email deleted!')
    } catch (err) {
      console.error('deleteEmail error:', err)
    }
  }

  return (
    <div>
      <h3>Flagged Emails</h3>

      {emails && emails.length > 0 ? (
        emails.map((email, idx) => (
          <div key={email.id ?? idx} className="email-card">
            <p><b>From:</b> {email.from ?? email.sender ?? 'Unknown'}</p>
            <p><b>Subject:</b> {email.subject}</p>
            <p>{email.snippet ?? email.preview ?? ''}</p>

            <p style={{ color: (email.is_phishing || (email.score || 0) >= 0.5) ? 'red' : 'green' }}>
              {(email.is_phishing || (email.score || 0) >= 0.5) ? '⚠️ Phishing' : '✔ Safe'}
            </p>

            {/* Action buttons for each email */}
            <button onClick={() => markSafe(email.id)}>Mark Safe</button>
            <button onClick={() => deleteEmail(email.id)}>Delete</button>

            <hr />
          </div>
        ))
      ) : (
        <p>No flagged emails found.</p>
      )}
    </div>
  )
}
