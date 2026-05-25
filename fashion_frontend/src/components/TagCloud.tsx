import { motion } from 'framer-motion'
import { useLanguage } from '../i18n/LanguageContext'
import type { Tag } from '../api'

interface Props {
  tags: Tag[]
}

export default function TagCloud({ tags }: Props) {
  const { t, translateAttr } = useLanguage()

  if (tags.length === 0) {
    return (
      <div className="tags-empty">
        <span>{t('tagsEmpty')}</span>
      </div>
    )
  }

  return (
    <div className="tag-cloud">
      {tags.map((t, i) => (
        <motion.span
          key={t.label}
          className="tag"
          initial={{ opacity: 0, scale: 0.8 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: i * 0.03 }}
          style={{
            opacity: 0.5 + t.confidence * 0.5,
            borderColor: `hsl(${200 + t.confidence * 60}, 50%, 60%)`,
          }}
        >
          <span className="tag-label">{translateAttr(t.label)}</span>
          <span className="tag-conf">{(t.confidence * 100).toFixed(0)}%</span>
        </motion.span>
      ))}
    </div>
  )
}
