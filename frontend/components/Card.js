export default function Card({ title, children }) {
  return (
    <div style={{
      background: '#fff',
      borderRadius: 8,
      padding: 16,
      boxShadow: '0 6px 18px rgba(2,6,23,0.06)',
      marginBottom: 12
    }}>
      <h3 style={{ margin: '0 0 8px 0' }}>{title}</h3>
      {children}
    </div>
  )
}
