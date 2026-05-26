import MetaBadges from './MetaBadges'
import styles from './ChatMessage.module.css'

export default function ChatMessage({ msg }) {
  const isUser = msg.role === 'user'

  return (
    <div className={`${styles.wrapper} ${isUser ? styles.user : styles.ai}`}>
      {!isUser && <div className={styles.avatar}>AI</div>}

      <div className={styles.content}>
        <div className={`${styles.bubble} ${isUser ? styles.userBubble : styles.aiBubble}`}>
          {msg.text}
        </div>

        {!isUser && msg.meta && (
          <MetaBadges
            intent={msg.meta.intent}
            sentiment={msg.meta.sentiment}
            confidence={msg.meta.intent_confidence}
            isUrgent={msg.meta.is_urgent}
          />
        )}
      </div>

      {isUser && <div className={`${styles.avatar} ${styles.userAvatar}`}>You</div>}
    </div>
  )
}
