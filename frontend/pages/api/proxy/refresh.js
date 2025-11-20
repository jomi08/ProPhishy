const BACKEND = process.env.BACKEND_URL || 'http://127.0.0.1:8000'

export default async function handler(req, res) {
  try {
    const r = await globalThis.fetch(`${BACKEND}/refresh`, { method: 'POST' })
    if (!r.ok) throw new Error('backend refresh failed')
    res.status(200).json({ ok: true })
  } catch (err) {
    console.error('proxy error', err)
    res.status(500).json({ error: 'proxy_error', detail: err.message })
  }
}
