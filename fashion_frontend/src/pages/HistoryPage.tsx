import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Trash2, ImageIcon, ChevronDown } from 'lucide-react'
import { useLanguage } from '../i18n/LanguageContext'
import { getHistory, removeFromHistory, clearHistory } from '../utils/history'
import type { HistoryItem } from '../api'
import ImageWithOverlay from '../components/ImageWithOverlay'
import TagCloud from '../components/TagCloud'
import CroppedGrid from '../components/CroppedGrid'

const COLORS = ['#d4a574', '#7eb8da', '#9fd5a1', '#e8a87c', '#c38d9e', '#85cdca']

export default function HistoryPage() {
  const { t, translateAttr } = useLanguage()
  const [history, setHistory] = useState<HistoryItem[]>([])
  const [expanded, setExpanded] = useState<string | null>(null)

  useEffect(() => {
    setHistory(getHistory())
  }, [])

  const handleRemove = (id: string) => {
    removeFromHistory(id)
    setHistory(getHistory())
  }

  const handleClear = () => {
    if (confirm(t('clearConfirm'))) {
      clearHistory()
      setHistory([])
    }
  }

  const formatDate = (ts: number) => {
    const d = new Date(ts)
    return d.toLocaleString('uk-UA', {
      day: '2-digit',
      month: '2-digit',
      year: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    })
  }

  return (
    <motion.div
      className="history-page"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
    >
      <div className="history-header">
        <h1>{t('historyTitle')}</h1>
        {history.length > 0 && (
          <button className="btn-clear" onClick={handleClear}>
            <Trash2 size={18} />
            {t('clearHistory')}
          </button>
        )}
      </div>

      {history.length === 0 ? (
        <div className="history-empty">
          <ImageIcon size={64} />
          <p>{t('historyEmpty')}</p>
          <span>{t('historyEmptyHint')}</span>
        </div>
      ) : (
        <div className="history-list">
          <AnimatePresence>
            {history.map((item) => (
              <motion.article
                key={item.id}
                className="history-card"
                layout
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, x: -100 }}
              >
                <div className="history-card-header">
                  <div className="history-meta">
                    <span className="history-date">{formatDate(item.timestamp)}</span>
                    {item.filename && (
                      <span className="history-filename">{item.filename}</span>
                    )}
                  </div>
                  <div className="history-actions">
                    <button
                      className="btn-expand"
                      onClick={() => setExpanded(expanded === item.id ? null : item.id)}
                    >
                      <ChevronDown
                        size={20}
                        style={{ transform: expanded === item.id ? 'rotate(180deg)' : 'none' }}
                      />
                    </button>
                    <button
                      className="btn-remove"
                      onClick={() => handleRemove(item.id)}
                    >
                      <Trash2 size={18} />
                    </button>
                  </div>
                </div>

                <div className="history-preview">
                  <img
                    src={`data:image/jpeg;base64,${item.image}`}
                    alt=""
                  />
                  <div className="history-tags-mini">
                    {item.tags.slice(0, 5).map((t) => (
                      <span key={t.label} className="mini-tag">{translateAttr(t.label)}</span>
                    ))}
                  </div>
                </div>

                <AnimatePresence>
                  {expanded === item.id && (
                    <motion.div
                      className="history-expanded"
                      initial={{ height: 0, opacity: 0 }}
                      animate={{ height: 'auto', opacity: 1 }}
                      exit={{ height: 0, opacity: 0 }}
                    >
                      {(() => {
                        const parts: string[] = []
                        if (item.detections.length > 0) {
                          const items = [...new Map(item.detections.map((d) => [d.label, d.confidence])).entries()]
                            .sort((a, b) => b[1] - a[1])
                            .slice(0, 8)
                            .map(([l, c]) => `${translateAttr(l)} (${(c * 100).toFixed(0)}%)`)
                          parts.push(`${t('summaryDetected')}: ${items.join(', ')}.`)
                        }
                        if (item.tags.length > 0) {
                          const topTags = item.tags.slice(0, 5).map((x) => translateAttr(x.label))
                          parts.push(`${t('summaryTags')}: ${topTags.join(', ')}.`)
                        }
                        const summary = parts.length > 0 ? parts.join(' ') : t('summaryDone')
                        return <p className="history-summary">{summary}</p>
                      })()}
                      <div className="history-detail-grid">
                        <div className="detail-block">
                          <h4>{t('withHighlight')}</h4>
                          <ImageWithOverlay
                            imageBase64={item.image}
                            detections={item.detections}
                            colors={COLORS}
                          />
                        </div>
                        <div className="detail-block">
                          <h4>{t('tags')}</h4>
                          <TagCloud tags={item.tags} />
                        </div>
                      </div>
                      {item.detections.length > 0 && (
                        <div className="detail-block">
                          <h4>{t('elements')}</h4>
                          <CroppedGrid detections={item.detections} colors={COLORS} />
                        </div>
                      )}
                    </motion.div>
                  )}
                </AnimatePresence>
              </motion.article>
            ))}
          </AnimatePresence>
        </div>
      )}
    </motion.div>
  )
}
