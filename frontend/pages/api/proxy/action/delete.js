const BACKEND = process.env.BACKEND_URL || 'http://127.0.0.1:8000'

export default async function handler(req, res) {
  try {
    const r = await globalThis.fetch(`${BACKEND}/action/delete`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: req.body ? JSON.stringify(req.body) : await req.text()
    })
    const json = await r.json()
    res.status(r.status).json(json)
  } catch (err) {
    console.error('proxy error', err)
    res.status(500).json({ error: 'proxy_error', detail: err.message })
  }
}
