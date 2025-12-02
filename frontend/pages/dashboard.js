import Metrics from '../components/Metrics'
import EmailList from '../components/EmailList'
import { useState, useEffect } from 'react'

export default function Dashboard() {
  const [emails, setEmails] = useState([])

  const fetchEmails = async () => {
    const res = await fetch('/api/proxy/emails')
    const data = await res.json()
    const emailsWithPhishingStatus = data.map(email => ({
      ...email,
      is_phishing: email.score > 0.5 // Assuming a threshold of 0.5
    }))
    setEmails(emailsWithPhishingStatus)
  }

  useEffect(() => {
    fetchEmails()
  }, [])

  return (
    <main className="container">
      <header>
        <h1>Dashboard</h1>
        <p className="muted">Live metrics and recent flagged emails</p>
      </header>

      <section style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: 16, marginTop: 18 }}>
        <div>
          <Metrics />
          <EmailList emails={emails} />
        </div>
        <aside>
          <div style={{ position: 'sticky', top: 16 }}>
            <div style={{ marginBottom: 12 }}>
              <h3>Quick actions</h3>
              <button onClick={() => { fetch('/api/proxy/refresh').then(() => alert('Refresh triggered')) }} style={{ padding: '8px 12px', borderRadius: 6, background: '#2563eb', color: '#fff', border: 'none' }}>Refresh now</button>
            </div>

            <div>
              <h3>Filters</h3>
              <p className="muted">(Not wired yet) Add date range or minimum score</p>
            </div>
          </div>
        </aside>
      </section>
    </main>
  )
}
