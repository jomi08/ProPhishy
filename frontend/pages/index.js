import useSWR from 'swr'
import Header from '../components/Header'
import Metrics from '../components/Metrics'
import EmailList from '../components/EmailList'

const fetcher = (url) => fetch(url).then((res) => res.json())

function Home() {
  const { data: status } = useSWR('/api/hello', fetcher)
  const { data: metrics } = useSWR('/api/proxy/metrics', fetcher)
  const { data: emails } = useSWR('/api/proxy/emails', fetcher)

  const refresh = async () => {
    await fetch('/api/proxy/refresh', { method: 'POST' })
  }

  return (
    <main className="container">
      <Header />

      <section>
        <h2>Backend status</h2>
        {!status && <p className="muted">Loading status...</p>}
        {status && <pre>{JSON.stringify(status, null, 2)}</pre>}
      </section>

      <section style={{ marginTop: 16 }}>
        <h2>Metrics</h2>
        {metrics ? <Metrics /> : <p className="muted">Loading metrics...</p>}
      </section>

      <section style={{ marginTop: 16 }}>
        <h2>Flagged emails</h2>
        {emails ? <EmailList /> : <p className="muted">Loading flagged emails...</p>}
      </section>

      <footer style={{ marginTop: 18 }}>
        <button onClick={refresh} style={{ padding: '8px 12px', borderRadius: 6, background: '#2563eb', color: '#fff', border: 'none' }}>Refresh</button>
      </footer>
    </main>
  )
}

export default Home
