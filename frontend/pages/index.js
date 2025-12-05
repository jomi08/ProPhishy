// frontend/src/index.js
import React, { useEffect, useMemo, useState } from 'react'
import useSWR from 'swr'

// Helper to determine API base safely in both server and client environments
function getApiBase() {
  try {
    if (typeof process !== 'undefined' && process?.env?.NEXT_PUBLIC_API_URL) {
      return process.env.NEXT_PUBLIC_API_URL
    }
  } catch (e) {}
  try {
    if (typeof window !== 'undefined') {
      // eslint-disable-next-line no-underscore-dangle
      const win = window
      if (win.__ENV__ && win.__ENV__.NEXT_PUBLIC_API_URL) return win.__ENV__.NEXT_PUBLIC_API_URL
    }
  } catch (e) {}
  return 'http://localhost:8000'
}

const API_BASE = getApiBase()
const fetcher = async (url) => {
  const res = await fetch(url)
  const text = await res.text()
  try {
    return text ? JSON.parse(text) : {}
  } catch (e) {
    return text
  }
}

function StatCard({ title, value, hint, pct, cardType }) {
  return (
    <div 
      className="stat-card p-4 rounded-2xl shadow-md w-full md:w-auto" 
      data-animate="true"
    >
      <div className="flex items-start gap-4">
        <div className="flex-1">
          <h4 className="stat-title text-sm font-medium" style={{ color: '#1B1F23' }}>{title}</h4>
          <div className="mt-2 text-2xl font-extrabold stat-value" style={{ color: '#1B1F23' }}>{value}</div>
          {hint ? <div className="text-xs mt-1" style={{ color: '#1B1F23', opacity: 0.7 }}>{hint}</div> : null}
          {typeof pct === 'number' && (
            <div className="mt-3">
              <div className="w-full rounded-full h-2 overflow-hidden" style={{ backgroundColor: 'rgba(0,0,0,0.1)' }}>
                <div
                  style={{ 
                    width: `${Math.max(0, Math.min(100, pct))}%`
                  }}
                  className="h-2 rounded-full stat-bar"
                />
              </div>
              <div className="text-xs mt-1" style={{ color: '#1B1F23', opacity: 0.7 }}>{pct.toFixed(0)}% of total</div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

function EmailRow({ e, onOpen }) {
  const score = (typeof e.score === 'number') ? e.score : parseFloat(e.score || 0)
  const isMarkedSafe = e.marked_safe === true
  const label = isMarkedSafe ? 'legit' : (e.label ? String(e.label).toLowerCase() : (score >= 0.5 ? 'spam' : 'legit'))
  const badgeClass = label === 'spam'
    ? 'px-2 py-1 text-xs rounded bg-red-600 text-white'
    : 'px-2 py-1 text-xs rounded bg-green-500 text-white flex items-center justify-center gap-1'

  return (
    <tr className="border-t hover:bg-gray-50">
      <td className="py-3 px-2 text-sm">{e.from}</td>
      <td className="py-3 px-2 text-sm">{e.subject}</td>
      <td className="py-3 px-2 text-sm">{(score || 0).toFixed(3)}</td>
      <td className="py-3 px-2 text-right flex items-center justify-end gap-2">
        <div className={badgeClass} style={{ minWidth: 56, textAlign: 'center' }}>
          {label === 'spam' ? 'Spam' : (
            <>
              <span>✓</span>
              <span>Safe</span>
            </>
          )}
        </div>

        <button onClick={() => onOpen(e)} className="px-2 py-1 text-xs rounded action-btn" type="button">View</button>
      </td>
    </tr>
  )
}

export default function Home() {
  // Use internal Next.js proxy endpoints to avoid CORS and env issues
  const METRICS_ENDPOINT = '/api/proxy/metrics'
  const EMAILS_ENDPOINT = '/api/proxy/emails'

  const { data: metrics, mutate: mutateMetrics, error: errMetrics } = useSWR(
    METRICS_ENDPOINT,
    fetcher,
    { refreshInterval: 0 }
  )

  const { data: emails, mutate: mutateEmails } = useSWR(EMAILS_ENDPOINT, fetcher, {
    refreshInterval: 0,
  })

  const [loadingAction, setLoadingAction] = useState(false)
  const [refreshing, setRefreshing] = useState(false)
  const [viewMode, setViewMode] = useState('table') // 'table' or 'cards'
  const [q, setQ] = useState('')
  const [selected, setSelected] = useState(null)
  const [lastRefreshed, setLastRefreshed] = useState(null)

  useEffect(() => {
    if (metrics || emails) setLastRefreshed(new Date())
  }, [metrics, emails])

  const filteredEmails = useMemo(() => {
    const list = Array.isArray(emails) ? emails : (emails?.items ?? emails?.emails ?? [])
    if (!list || !Array.isArray(list)) return []
    if (!q) return list
    const s = q.trim().toLowerCase()
    return list.filter((e) => {
      return (
        (e.from || '').toLowerCase().includes(s) ||
        (e.subject || '').toLowerCase().includes(s) ||
        String(e.score || '').toLowerCase().includes(s)
      )
    })
  }, [emails, q])

  const total = metrics?.total ?? 0
  const safe = metrics?.safe ?? metrics?.read ?? 0
  const spam = metrics?.spam ?? 0

  async function refreshAll() {
    setRefreshing(true)
    setLastRefreshed(new Date())
    try {
      // trigger backend refresh via the proxy so new messages are fetched and classified
      const res = await fetch('/api/proxy/refresh', { method: 'POST' })
      const data = await res.json()
      await Promise.all([mutateMetrics(), mutateEmails()])
      if (data && data.total !== undefined) {
        alert(`Refreshed! Total: ${data.total}, Spam: ${data.spam}, Safe: ${data.safe}`)
      }
    } catch (err) {
      // ignore refresh errors but still revalidate cached endpoints
      console.warn('refresh trigger failed', err)
      await Promise.all([mutateMetrics(), mutateEmails()])
    } finally {
      setRefreshing(false)
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-50 to-white p-6">
      <style jsx global>{`
        /* (kept your styling unchanged) */
        :root { --maroon-950:#2a0606; --maroon-900:#3b0b0b; --maroon-800:#5a0f12; --maroon-700:#7a1421; --maroon-600:#8f1b2a; --maroon-500:#b02a37; --maroon-300:#e07b85; --muted:#6b7280; --surface: rgba(255,255,255,0.06); }
        body { font-family: "Montserrat", "Poppins", Inter, ui-sans-serif; font-size:22px; color:#0f172a; background-color: #4a0d12; background-image: radial-gradient(rgba(255,255,255,0.02) 1px, transparent 1px), linear-gradient(180deg, rgba(74,13,18,0.65), rgba(30,4,6,0.75)); background-size: 12px 12px, 100% 100%; }
        .panel, .stat-card, .modal > div, .bg-white { background: linear-gradient(180deg, rgba(255,235,238,0.82), rgba(255,220,225,0.78)); border: 1px solid rgba(255,255,255,0.22); box-shadow: 0 8px 24px rgba(50, 10, 10, 0.20); backdrop-filter: blur(8px) saturate(140%); -webkit-backdrop-filter: blur(8px) saturate(140%); }
        .stat-card { transition: transform 280ms cubic-bezier(.2,.9,.2,1), box-shadow 280ms; transform-origin: center; will-change: transform; }
        .stat-card:hover { transform: translateY(-6px) scale(1.02); box-shadow: 0 18px 40px rgba(59,11,11,0.14); }
        .stat-value { font-size: 1.6rem; color: var(--maroon-950); text-shadow: 0 1px 0 rgba(255,255,255,0.6); }
        .stat-title { color: var(--muted); font-size: 0.95rem; }
        .stat-bar { background: linear-gradient(90deg, var(--maroon-800), var(--maroon-500)); transition: width 700ms cubic-bezier(.2,.9,.2,1); }
        .refresh-btn { background: linear-gradient(180deg, var(--maroon-800), var(--maroon-600)); color: white; border-radius: 10px; padding: 8px 12px; box-shadow: 0 6px 18px rgba(176,42,55,0.18); transition: transform 150ms, box-shadow 150ms; }
        .action-btn { background: rgba(255,255,255,0.9); color: var(--maroon-900); border: 1px solid rgba(176,42,55,0.06); border-radius: 8px; transition: transform 150ms; }
        .action-btn:hover { transform: translateY(-2px); }
        .text-muted { color: var(--muted); }
        @media (max-width: 768px) { body { font-size: 16px; } .stat-value { font-size: 1.2rem; } }
      `}</style>

      <div className="max-w-6xl mx-auto">
        <header className="mb-6 flex items-center justify-between">
          <div>
            <h1 className="text-6xl font-extrabold tracking-tight" style={{fontFamily:'"Dancing Script", cursive', letterSpacing:'2px', color:'var(--maroon-900)'}}>ProPhishy</h1>
            <p className="text-muted mt-1">Phishing detector admin dashboard</p>
          </div>

          <div className="flex items-center gap-3">
            <div className="text-sm text-muted text-right">
              <div>Backend: {errMetrics ? 'Error' : metrics ? 'Connected' : 'Loading...'}</div>
              <div className="text-xs text-gray-400">{lastRefreshed ? lastRefreshed.toLocaleTimeString() : ''}</div>
            </div>
            <button
              onClick={refreshAll}
              className="px-3 py-2 refresh-btn rounded-lg shadow flex items-center gap-2"
              type="button"
              disabled={refreshing}
            >
              {refreshing ? (
                <>
                  <span className="inline-block animate-spin">⟳</span>
                  Refreshing...
                </>
              ) : (
                '⟳ Refresh'
              )}
            </button>
          </div>
        </header>

        <section className="grid gap-6 md:grid-cols-3 mb-6">
          <StatCard
            title="Total emails"
            value={total ?? '—'}
            hint="All emails received"
            pct={total ? 100 : 0}
            cardType="total"
          />
          <StatCard
            title="Safe emails"
            value={safe ?? '—'}
            hint="Marked as safe"
            pct={total ? ((safe || 0) / total) * 100 : 0}
            cardType="safe"
          />
          <StatCard
            title="Spam / Flagged"
            value={spam ?? '—'}
            hint="Flagged by the classifier"
            pct={total ? ((spam || 0) / total) * 100 : 0}
            cardType="spam"
          />
        </section>

        <section className="mb-6 panel rounded-2xl shadow p-4">
          <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-3">
            <div className="flex items-center gap-3">
              <input
                value={q}
                onChange={(e) => setQ(e.target.value)}
                placeholder="Search flagged: from, subject, score"
                className="px-3 py-2 border rounded-lg w-72 text-sm"
              />
              <div className="flex items-center gap-2">
                <button 
                  onClick={() => setViewMode('table')} 
                  className={`px-3 py-2 rounded-lg text-sm transition-all ${
                    viewMode === 'table' 
                      ? 'border-2 border-gray-400 shadow-md font-bold' 
                      : 'bg-gray-100 border-2 border-transparent hover:border-gray-300'
                  }`}
                  style={viewMode === 'table' ? { color: '#1B1F23' } : {}}
                  type="button"
                >
                  Table
                </button>
                <button 
                  onClick={() => setViewMode('cards')} 
                  className={`px-3 py-2 rounded-lg text-sm transition-all ${
                    viewMode === 'cards' 
                      ? 'border-2 border-gray-400 shadow-md font-bold' 
                      : 'bg-gray-100 border-2 border-transparent hover:border-gray-300'
                  }`}
                  style={viewMode === 'cards' ? { color: '#1B1F23' } : {}}
                  type="button"
                >
                  Cards
                </button>
              </div>
            </div>
            <div className="text-sm text-muted">Showing <strong>{filteredEmails?.length ?? 0}</strong> emails</div>
          </div>
        </section>

        <section className="space-y-4">
          {viewMode === 'table' ? (
            <div className="panel rounded-2xl overflow-hidden">
              <table className="w-full">
                <thead>
                  <tr className="text-xs text-gray-400 border-b">
                    <th className="text-left p-3">From</th>
                    <th className="text-left p-3">Subject</th>
                    <th className="text-left p-3">Score</th>
                    <th className="text-right p-3">Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {filteredEmails.length === 0 ? (
                    <tr>
                      <td colSpan={4} className="p-6 text-center text-muted">No emails found</td>
                    </tr>
                  ) : (
                    filteredEmails.map((e) => (
                      <EmailRow
                        key={e.id ?? e._id ?? Math.random()}
                        e={e}
                        onOpen={(item) => setSelected(item)}
                      />
                    ))
                  )}
                </tbody>
              </table>
            </div>
          ) : (
            <div className="grid md:grid-cols-3 gap-4">
              {filteredEmails.length === 0 ? (
                <div className="text-muted">No emails found</div>
              ) : (
                filteredEmails.map((e) => {
                  const isMarkedSafe = e.marked_safe === true
                  const isSpam = isMarkedSafe ? false : (e.label === 'spam' || (e.label == null && (e.score || 0) >= 0.5))
                  
                  return (
                    <div key={e.id ?? e._id ?? Math.random()} className="bg-white p-4 rounded-2xl shadow">
                      <div className="font-semibold" style={{ color: 'var(--maroon-700)', fontSize: '1rem' }}>{e.from}</div>
                      <div className="text-sm text-gray-600 mt-1">{e.subject}</div>

                      <div className="mt-3 flex items-center justify-between">
                        <div>
                          <div className="text-xs text-muted">Score: {(e.score ?? 0).toFixed(3)}</div>
                          <div style={{ marginTop: 6 }}>
                            <span className={isSpam
                              ? 'px-2 py-1 rounded bg-red-600 text-white text-xs'
                              : 'px-2 py-1 rounded bg-green-500 text-white text-xs inline-flex items-center gap-1'}>
                              {isSpam ? 'Spam' : (
                                <>
                                  <span>✓</span>
                                  <span>Safe</span>
                                </>
                              )}
                            </span>
                          </div>
                        </div>
                        <div className="flex items-center gap-2">
                          <button onClick={() => setSelected(e)} className="px-2 py-1 rounded action-btn text-sm">View</button>
                        </div>
                      </div>

                    </div>
                  )
                })
              )}
            </div>
          )}

          <div className="text-right">
            <small className="text-muted">Actions: {loadingAction ? 'processing...' : 'idle'}</small>
          </div>
        </section>

        {/* Modal */}
        {selected && (() => {
          const isMarkedSafe = selected.marked_safe === true
          const isSpam = isMarkedSafe ? false : (selected.label === 'spam' || (selected.label == null && (selected.score || 0) >= 0.5))
          
          return (
            <div className="fixed inset-0 z-40 flex items-center justify-center modal" role="dialog">
              <div className="bg-white max-w-2xl w-full rounded-2xl shadow p-6">
                <div>
                  <h3 className="text-lg font-semibold">Email from {selected.from}</h3>
                  <div className="text-sm text-muted mt-1">Subject: {selected.subject}</div>
                  <div className="mt-2">
                    <span className={isSpam
                      ? 'px-2 py-1 rounded bg-red-600 text-white text-xs'
                      : 'px-2 py-1 rounded bg-green-500 text-white text-xs flex items-center gap-1 inline-flex'}>
                      {isSpam ? 'Spam' : (
                        <>
                          <span>✓</span>
                          <span>Safe</span>
                        </>
                      )}
                    </span>
                    <span className="text-xs text-muted ml-2">Score: {(selected.score ?? 0).toFixed(3)}</span>
                  </div>
                </div>

                <div className="mt-4">
                  <h4 className="text-sm text-muted">Preview</h4>
                  <div className="mt-2 p-4 bg-surface rounded overflow-y-auto" style={{ maxHeight: "300px", whiteSpace: "pre-wrap", fontSize: "14px" }}>
                    {selected.body ?? selected.preview ?? 'No preview available'}
                  </div>
                </div>

                <div className="mt-4 flex justify-end gap-2">
                  {isSpam && (
                  <button
                    onClick={async () => {
                      setLoadingAction(true)
                      try {
                        const res = await fetch('/api/proxy/action/mark-safe', {
                          method: 'POST',
                          headers: { 'Content-Type': 'application/json' },
                          body: JSON.stringify({ id: selected.id })
                        })
                        if (res.ok) {
                          alert('Marked as safe!')
                          // Update the selected email's label to show it as safe
                          setSelected({ ...selected, label: 'legit', marked_safe: true })
                          await mutateEmails()
                          await mutateMetrics()
                        }
                      } catch (err) {
                        console.error('mark-safe failed', err)
                      } finally {
                        setLoadingAction(false)
                      }
                    }}
                    className="px-3 py-1.5 rounded bg-green-500 text-white hover:bg-green-600"
                    style={{ fontSize: '0.9rem' }}
                    disabled={loadingAction}
                  >
                    {loadingAction ? 'Processing...' : 'Mark Safe'}
                    </button>
                  )}
                  <button onClick={() => setSelected(null)} className="px-3 py-1.5 rounded action-btn" style={{ fontSize: '0.9rem' }}>Close</button>
                </div>
              </div>
            </div>
          )
        })()}
      </div>
    </div>
  )
}
