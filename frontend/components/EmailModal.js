export default function EmailModal({ email, onClose, onMarkSafe, onDelete }) {
  if (!email) return null

  return (
    <div style={{ position: 'fixed', inset: 0, background: 'rgba(2,6,23,0.5)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
      <div style={{ width: '90%', maxWidth: 880, background: '#fff', borderRadius: 8, padding: 16 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div>
            <h2 style={{ margin: 0 }}>{email.subject || 'No subject'}</h2>
            <div className="muted">From: {email.from}</div>
          </div>
          <div>
            <button onClick={onMarkSafe} style={{ marginRight: 8, padding: '6px 10px' }}>Mark Safe</button>
            <button onClick={onDelete} style={{ marginRight: 8, padding: '6px 10px', background: '#ef4444', color: '#fff' }}>Delete</button>
            <button onClick={onClose} style={{ padding: '6px 10px' }}>Close</button>
          </div>
        </div>

        <hr />
        <div style={{ maxHeight: '60vh', overflow: 'auto', whiteSpace: 'pre-wrap' }}>
          {email.body || email.snippet || '(no preview)'}
        </div>
      </div>
    </div>
  )
}
