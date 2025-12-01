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

function StatCard({ title, value, hint, pct, onClick }) {
  return (
    <div className="stat-card p-4 bg-white rounded-2xl shadow-md w-full md:w-auto">
      <div className="flex items-start gap-4">
        <div className="flex-1">
          <h4 className="stat-title text-sm font-medium">{title}</h4>
          <div className="mt-2 text-2xl font-extrabold stat-value">{value}</div>
          {hint ? <div className="text-xs text-muted mt-1">{hint}</div> : null}
          {typeof pct === 'number' && (
            <div className="mt-3">
              <div className="w-full bg-surface rounded-full h-2 overflow-hidden">
                <div
                  style={{ width: `${Math.max(0, Math.min(100, pct))}%` }}
                  className="h-2 rounded-full stat-bar"
                />
              </div>
              <div className="text-xs text-muted mt-1">{pct.toFixed(0)}% of total</div>
            </div>
          )}
        </div>
        <div className="flex-shrink-0">
          <button
            onClick={onClick}
            className="px-3 py-1 refresh-btn text-sm"
            type="button"
          >
            Refresh
          </button>
        </div>
      </div>
    </div>
  )
}

function EmailRow({ e, onOpen, onMarkSafe, onDelete }) {
  return (
    <tr className="border-t hover:bg-gray-50">
      <td className="py-3 px-2 text-sm">{e.from}</td>
      <td className="py-3 px-2 text-sm">{e.subject}</td>
      <td className="py-3 px-2 text-sm">{e.score ?? '—'}</td>
      <td className="py-3 px-2 text-right">
        <button
          onClick={() => onOpen(e)}
          className="mr-2 px-2 py-1 rounded action-btn"
          type="button"
        >
          View
        </button>
        <button
          onClick={() => onMarkSafe(e.id ?? e._id)}
          className="mr-2 px-2 py-1 rounded safe-btn"
          type="button"
        >
          Safe
        </button>
        <button
          onClick={() => onDelete(e.id ?? e._id)}
          className="px-2 py-1 rounded delete-btn"
          type="button"
        >
          Delete
        </button>
      </td>
    </tr>
  )
}

export default function Home() {
  const { data: metrics, mutate: mutateMetrics, error: errMetrics } = useSWR(
    `${API_BASE}/metrics`,
    fetcher,
    { refreshInterval: 0 }
  )
  const { data: flagged, mutate: mutateFlagged } = useSWR(`${API_BASE}/flagged`, fetcher, {
    refreshInterval: 0,
  })

  const [loadingAction, setLoadingAction] = useState(false)
  const [viewMode, setViewMode] = useState('table') // 'table' or 'cards'
  const [q, setQ] = useState('')
  const [selected, setSelected] = useState(null)
  const [lastRefreshed, setLastRefreshed] = useState(null)

  useEffect(() => {
    if (metrics || flagged) setLastRefreshed(new Date())
  }, [metrics, flagged])

  const emails = useMemo(() => {
    const list = Array.isArray(flagged) ? flagged : flagged?.items ?? flagged?.emails ?? []
    if (!q) return list
    const s = q.trim().toLowerCase()
    return list.filter((e) => {
      return (
        (e.from || '').toLowerCase().includes(s) ||
        (e.subject || '').toLowerCase().includes(s) ||
        String(e.score || '').toLowerCase().includes(s)
      )
    })
  }, [flagged, q])

  const total = metrics && (metrics.total ?? metrics.total_emails ?? 0)
  const read = metrics && (metrics.read ?? metrics.read_emails ?? 0)
  const spam = metrics && (metrics.spam ?? metrics.flagged_count ?? 0)

  async function refreshAll() {
    setLastRefreshed(new Date())
    await Promise.all([mutateMetrics(), mutateFlagged()])
  }

  async function markSafe(id) {
    if (!id) return
    setLoadingAction(true)
    try {
      await fetch(`${API_BASE}/action/mark-safe`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ id }),
      })
      await mutateFlagged()
    } catch (e) {
      // eslint-disable-next-line no-console
      console.error('markSafe failed', e)
    }
    setLoadingAction(false)
  }

  async function deleteEmail(id) {
    if (!id) return
    setLoadingAction(true)
    try {
      await fetch(`${API_BASE}/action/delete`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ id }),
      })
      await mutateFlagged()
    } catch (e) {
      // eslint-disable-next-line no-console
      console.error('deleteEmail failed', e)
    }
    setLoadingAction(false)
  }

  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-50 to-white p-6">
      <style jsx global>{`
        /* Maroon-themed color variables */
        :root { --maroon-950:#2a0606; --maroon-900:#3b0b0b; --maroon-800:#5a0f12; --maroon-700:#7a1421; --maroon-600:#8f1b2a; --maroon-500:#b02a37; --maroon-300:#e07b85; --muted:#6b7280; --surface: rgba(255,255,255,0.06); }

        /* Page background: subtle maroon texture */
        body { 
          font-family: "Montserrat", "Poppins", Inter, ui-sans-serif; 
          font-size:22px; 
          color:#0f172a;
          /* layered subtle texture + gradient */
          background-color: #4a0d12; /* deep maroon */
          background-image: radial-gradient(rgba(255,255,255,0.02) 1px, transparent 1px), linear-gradient(180deg, rgba(74,13,18,0.65), rgba(30,4,6,0.75));
          background-size: 12px 12px, 100% 100%;
        }

        /* Glass-morphism panels */
        .panel, .stat-card, .modal > div, .bg-white {
          background: linear-gradient(180deg, rgba(255,235,238,0.82), rgba(255,220,225,0.78));
          border: 1px solid rgba(255,255,255,0.22);
          box-shadow: 0 8px 24px rgba(50, 10, 10, 0.20);
          backdrop-filter: blur(8px) saturate(140%);
          -webkit-backdrop-filter: blur(8px) saturate(140%);
        }

        /* Stat card visuals + animated entrance */
        .stat-card { 
          transition: transform 280ms cubic-bezier(.2,.9,.2,1), box-shadow 280ms;
          transform-origin: center;
          will-change: transform;
        }
        .stat-card:hover { 
          transform: translateY(-6px) scale(1.02);
          box-shadow: 0 18px 40px rgba(59,11,11,0.14);
        }

        /* Floating animation for cards (subtle) */
        @keyframes floaty { 
          0% { transform: translateY(0px); }
          50% { transform: translateY(-4px); }
          100% { transform: translateY(0px); }
        }
        .stat-card[data-animate="true"] { animation: floaty 6s ease-in-out infinite; }

        /* Stat value emphasis */
        .stat-value { font-size: 1.6rem; color: var(--maroon-950); text-shadow: 0 1px 0 rgba(255,255,255,0.6); }
        .stat-title { color: var(--muted); font-size: 0.95rem; }

        /* Progress bar */
        .stat-bar { background: linear-gradient(90deg, var(--maroon-800), var(--maroon-500)); transition: width 700ms cubic-bezier(.2,.9,.2,1); }
        .wobble { transition: transform 300ms; }

        /* Buttons with animated hover */
        .refresh-btn { background: linear-gradient(180deg, var(--maroon-800), var(--maroon-600)); color: white; border-radius: 10px; padding: 8px 12px; box-shadow: 0 6px 18px rgba(176,42,55,0.18); transition: transform 150ms, box-shadow 150ms; }
        .refresh-btn:hover { transform: translateY(-3px) scale(1.02); box-shadow: 0 22px 40px rgba(176,42,55,0.18); }
        .action-btn { background: rgba(255,255,255,0.9); color: var(--maroon-900); border: 1px solid rgba(176,42,55,0.06); border-radius: 8px; transition: transform 150ms; }
        .action-btn:hover { transform: translateY(-2px); }
        .safe-btn { background: linear-gradient(180deg,#ef9a9a,#ef6c6c); color: white; transition: transform 150ms; }
        .safe-btn:hover { transform: translateY(-2px); }
        .delete-btn { background: linear-gradient(180deg,var(--maroon-500),var(--maroon-700)); color: white; transition: transform 150ms; }
        .delete-btn:hover { transform: translateY(-2px) scale(1.02); }

        /* Table row hover and animated reveal */
        tbody tr { transition: background 200ms, transform 200ms; }
        tbody tr:hover { background: rgba(255,255,255,0.06); transform: translateX(4px); }

        /* Card view animations */
        .grid > div { transition: transform 350ms, box-shadow 350ms, opacity 300ms; }
        .grid > div:hover { transform: translateY(-6px); box-shadow: 0 18px 40px rgba(59,11,11,0.12); }

        /* Modal styling + entrance */
        .modal { display: flex; align-items: center; justify-content: center; }
        .modal > div { transform: translateY(12px); opacity: 0; animation: modalIn 260ms cubic-bezier(.2,.9,.2,1) forwards; }
        @keyframes modalIn { to { transform: translateY(0); opacity: 1; } }

        /* small UI tweaks */
        .text-muted { color: var(--muted); }

        /* Responsiveness tweaks */
        @media (max-width: 768px) {
          body { font-size: 16px; }
          .stat-value { font-size: 1.2rem; }
        }
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
              className="px-3 py-2 refresh-btn rounded-lg shadow"
              type="button"
            >
              Refresh
            </button>
          </div>
        </header>

        <section className="grid gap-6 md:grid-cols-3 mb-6">
          <StatCard
            title="Total emails"
            value={total ?? '—'}
            hint="All emails received"
            pct={total ? 100 : 0}
            onClick={() => mutateMetrics()}
          />

          <StatCard
            title="Read emails"
            value={read ?? '—'}
            hint="Marked as read"
            pct={total ? ((read || 0) / total) * 100 : 0}
            onClick={() => mutateMetrics()}
          />

          <StatCard
            title="Spam / Flagged"
            value={spam ?? '—'}
            hint="Flagged by the classifier"
            pct={total ? ((spam || 0) / total) * 100 : 0}
            onClick={() => mutateMetrics()}
          />
        </section>

        <section className="mb-6 panel rounded-2xl shadow p-4">
          <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-3">
            <div className="flex items-center gap-3">
              <input
                value={q}
                onChange={(e) => setQ(e.target.value)}
                placeholder="Search flagged: from, subject, score"
                className="px-3 py-2 border rounded-lg w-72"
              />

              <div className="flex items-center gap-2">
                <button
                  onClick={() => setViewMode('table')}
                  className={`px-3 py-2 rounded-lg ${viewMode === 'table' ? 'bg-maroon-primary text-white' : 'bg-gray-100'}`}
                  type="button"
                >
                  Table
                </button>
                <button
                  onClick={() => setViewMode('cards')}
                  className={`px-3 py-2 rounded-lg ${viewMode === 'cards' ? 'bg-maroon-primary text-white' : 'bg-gray-100'}`}
                  type="button"
                >
                  Cards
                </button>
              </div>
            </div>

            <div className="text-sm text-muted">Showing <strong>{emails.length}</strong> flagged emails</div>
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
                  {emails.length === 0 ? (
                    <tr>
                      <td colSpan={4} className="p-6 text-center text-muted">No flagged emails</td>
                    </tr>
                  ) : (
                    emails.map((e) => (
                      <EmailRow
                        key={e.id ?? e._id ?? Math.random()}
                        e={e}
                        onOpen={(item) => setSelected(item)}
                        onMarkSafe={(id) => markSafe(id)}
                        onDelete={(id) => deleteEmail(id)}
                      />
                    ))
                  )}
                </tbody>
              </table>
            </div>
          ) : (
            <div className="grid md:grid-cols-3 gap-4">
              {emails.length === 0 ? (
                <div className="text-muted">No flagged emails</div>
              ) : (
                emails.map((e) => (
                  <div key={e.id ?? e._id ?? Math.random()} className="bg-white p-4 rounded-2xl shadow">
                    <div className="font-semibold" style={{color: 'var(--maroon-700)'}}>{e.from}</div>
                    <div className="text-sm text-gray-600 mt-1">{e.subject}</div>
                    <div className="mt-3 flex items-center justify-between">
                      <div className="text-xs text-muted">Score: {e.score ?? '—'}</div>
                      <div className="flex items-center gap-2">
                        <button onClick={() => setSelected(e)} className="px-2 py-1 rounded action-btn text-sm">View</button>
                        <button onClick={() => markSafe(e.id ?? e._id)} className="px-2 py-1 rounded safe-btn text-sm">Safe</button>
                      </div>
                    </div>
                  </div>
                ))
              )}
            </div>
          )}

          <div className="text-right">
            <small className="text-muted">Actions: {loadingAction ? 'processing...' : 'idle'}</small>
          </div>
        </section>

        {/* Modal */}
        {selected && (
          <div className="fixed inset-0 z-40 flex items-center justify-center modal" role="dialog">
            <div className="bg-white max-w-2xl w-full rounded-2xl shadow p-6">
              <div className="flex items-start justify-between">
                <div>
                  <h3 className="text-lg font-semibold">Email from {selected.from}</h3>
                  <div className="text-sm text-muted mt-1">Subject: {selected.subject}</div>
                </div>
                <div>
                  <button onClick={() => setSelected(null)} className="px-3 py-1 rounded action-btn">Close</button>
                </div>
              </div>

              <div className="mt-4">
                <h4 className="text-sm text-muted">Preview</h4>
                <div className="mt-2 p-4 bg-surface rounded">{selected.body ?? selected.preview ?? 'No preview available'}</div>
              </div>

              <div className="mt-4 flex justify-end gap-2">
                <button onClick={() => { markSafe(selected.id ?? selected._id); setSelected(null) }} className="px-4 py-2 rounded safe-btn">Mark safe</button>
                <button onClick={() => { deleteEmail(selected.id ?? selected._id); setSelected(null) }} className="px-4 py-2 rounded delete-btn">Delete</button>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
