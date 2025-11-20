const BACKEND = process.env.BACKEND_URL || 'http://127.0.0.1:8000'

export default async function handler(req, res) {
  try {
    // Use the global fetch provided by Next/Node instead of importing node-fetch
    const r = await globalThis.fetch(`${BACKEND}/metrics`)
    const json = await r.json()
    res.status(200).json(json)
  } catch (err) {
    console.error('proxy error', err)
    res.status(500).json({ error: 'proxy_error', detail: err.message })
  }
}
