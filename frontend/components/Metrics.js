import useSWR from 'swr'
import Card from './Card'

const fetcher = (url) => fetch(url).then(r => r.json())

export default function Metrics() {
  const { data, error } = useSWR('/api/proxy/metrics', fetcher, { refreshInterval: 5000 })

  if (error) return <Card title="Metrics"><p className="muted">Failed to load metrics</p></Card>
  if (!data) return <Card title="Metrics"><p className="muted">Loading...</p></Card>

  return (
    <Card title="Metrics">
      <div style={{ display: 'flex', gap: 12 }}>
        <div style={{ flex: 1 }}>
          <strong style={{ fontSize: 28 }}>{data.flagged_count}</strong>
          <div className="muted">Flagged emails</div>
        </div>
        <div style={{ flex: 1 }}>
          <strong style={{ fontSize: 28 }}>{data.total_checked}</strong>
          <div className="muted">Emails checked</div>
        </div>
        <div style={{ flex: 1 }}>
          <strong style={{ fontSize: 28 }}>{data.today_flagged}</strong>
          <div className="muted">Flagged today</div>
        </div>
      </div>
    </Card>
  )
}
