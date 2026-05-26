import styles from './MetaBadges.module.css'

const SENTIMENT_COLOR = { positive: 'positive', negative: 'negative', neutral: 'neutral' }
const INTENT_LABEL = {
  payment_issue: 'Payment Issue',
  refund_query: 'Refund Query',
  account_access: 'Account Access',
  order_tracking: 'Order Tracking',
  technical_problem: 'Technical Problem',
  general_inquiry: 'General Inquiry',
}

export default function MetaBadges({ intent, sentiment, confidence, isUrgent }) {
  return (
    <div className={styles.row}>
      <span className={styles.badge}>{INTENT_LABEL[intent] ?? intent}</span>
      <span className={`${styles.badge} ${styles[SENTIMENT_COLOR[sentiment]]}`}>
        {sentiment}
      </span>
      <span className={styles.badge}>{(confidence * 100).toFixed(0)}% confident</span>
      {isUrgent && <span className={`${styles.badge} ${styles.urgent}`}>⚠ Urgent</span>}
    </div>
  )
}
