import { useState, useRef, useEffect } from 'react'
import ChatMessage from './components/ChatMessage'
import TypingIndicator from './components/TypingIndicator'
import { sendMessage } from './utils/api'
import styles from './App.module.css'

const SUGGESTIONS = [
  'My payment failed but money got deducted',
  'I want a refund for my cancelled order',
  'Cannot login to my account',
  'Where is my order?',
]

const WELCOME = {
  role: 'ai',
  text: "Hi! I'm your AI Support Assistant. I can help with payments, refunds, account access, order tracking, and more. How can I help you today?",
  id: 0,
}

export default function App() {
  const [messages, setMessages] = useState([WELCOME])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const bottomRef = useRef(null)
  const inputRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  async function handleSend(text) {
    const message = (text ?? input).trim()
    if (!message || loading) return

    setInput('')
    setMessages(prev => [...prev, { role: 'user', text: message, id: Date.now() }])
    setLoading(true)

    try {
      const data = await sendMessage(message)
      setMessages(prev => [
        ...prev,
        { role: 'ai', text: data.response, meta: data, id: Date.now() + 1 },
      ])
    } catch {
      setMessages(prev => [
        ...prev,
        { role: 'ai', text: 'Sorry, I could not connect to the server. Please try again.', id: Date.now() + 1 },
      ])
    } finally {
      setLoading(false)
      inputRef.current?.focus()
    }
  }

  function handleKey(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  return (
    <div className={styles.shell}>
      {/* Header */}
      <header className={styles.header}>
        <div className={styles.headerLeft}>
          <div className={styles.logo}>AI</div>
          <div>
            <div className={styles.title}>Support Assistant</div>
            <div className={styles.subtitle}>Powered by NLP · Always online</div>
          </div>
        </div>
        <div className={styles.statusDot} title="Online" />
      </header>

      {/* Messages */}
      <main className={styles.messages}>
        {messages.map(msg => (
          <ChatMessage key={msg.id} msg={msg} />
        ))}
        {loading && <TypingIndicator />}
        <div ref={bottomRef} />
      </main>

      {/* Suggestions */}
      {messages.length === 1 && (
        <div className={styles.suggestions}>
          {SUGGESTIONS.map(s => (
            <button key={s} className={styles.chip} onClick={() => handleSend(s)}>
              {s}
            </button>
          ))}
        </div>
      )}

      {/* Input */}
      <footer className={styles.inputArea}>
        <textarea
          ref={inputRef}
          className={styles.input}
          rows={1}
          placeholder="Type your message..."
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={handleKey}
          disabled={loading}
        />
        <button
          className={styles.sendBtn}
          onClick={() => handleSend()}
          disabled={!input.trim() || loading}
          aria-label="Send"
        >
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
            <line x1="22" y1="2" x2="11" y2="13" />
            <polygon points="22 2 15 22 11 13 2 9 22 2" />
          </svg>
        </button>
      </footer>
    </div>
  )
}
