import { useEffect, useState } from 'react'
import { motion } from 'framer-motion'
import { Search, Trash2 } from 'lucide-react'
import { fetchJob, resetJob, type CatalogItem, type CatalogState } from '../api'
import { useLanguage } from '../i18n/LanguageContext'
import { clearActiveJobId, getActiveJobId } from '../utils/jobs'

const TOKEN_SYNONYMS: Record<string, string> = {
  black: 'black',
  чорний: 'black',
  white: 'white',
  білий: 'white',
  red: 'red',
  червоний: 'red',
  blue: 'blue',
  синій: 'blue',
  green: 'green',
  зелений: 'green',
  dress: 'dress',
  сукня: 'dress',
  shirt: 'shirt',
  сорочка: 'shirt',
  blouse: 'shirt',
  blousee: 'shirt',
  dot: 'dot',
  dots: 'dot',
  polka: 'dot',
  горошок: 'dot',
  stripe: 'stripe',
  смужка: 'stripe',
  floral: 'floral',
  квітковий: 'floral',
  geometric: 'geometric',
  геометричний: 'geometric',
  vneck: 'v-neck',
  'v-neck': 'v-neck',
}

function normalizeToken(value: string): string {
  const normalized = value
    .toLowerCase()
    .replace(/[^a-zа-яіїєґ0-9]+/gi, ' ')
    .trim()
    .replace(/\s+/g, ' ')
  return TOKEN_SYNONYMS[normalized] ?? normalized
}

function tokenize(value: string): string[] {
  return value
    .split(/\s+/)
    .map((token) => normalizeToken(token))
    .filter(Boolean)
}

function buildItemSearchScore(item: CatalogItem, query: string, translateAttr: (label: string) => string): number {
  const tokens = tokenize(query)
  if (tokens.length === 0) return 0

  let score = 0
  const weightedLabels: Array<[string, number]> = [
    ...item.tags.map((tag) => [tag.label, tag.confidence * 1.1] as [string, number]),
    ...item.detections.map((detection) => [detection.label, detection.confidence * 1.25] as [string, number]),
    ...item.detections.flatMap((detection) =>
      (detection.attributes ?? []).map((attr) => [attr.label, attr.confidence] as [string, number])
    ),
  ]

  const normalizedPool = new Map<string, number>()
  for (const [label, confidence] of weightedLabels) {
    for (const variant of [label, translateAttr(label)]) {
      for (const token of tokenize(variant)) {
        normalizedPool.set(token, Math.max(normalizedPool.get(token) ?? 0, confidence))
      }
    }
  }

  for (const token of tokens) {
    let tokenScore = normalizedPool.get(token) ?? 0
    const textPool = [
      item.name,
      item.name_ua,
      item.summary,
      ...item.params.map((param) => `${param.name} ${param.value}`),
    ]
    if (textPool.some((value) => tokenize(value).includes(token))) {
      tokenScore = Math.max(tokenScore, 0.68)
    }
    score += tokenScore
  }

  return Math.min(100, Math.round((score / tokens.length) * 100))
}

export default function CatalogPage() {
  const { t, translateAttr } = useLanguage()
  const [catalog, setCatalog] = useState<CatalogState | null>(null)
  const [search, setSearch] = useState('')
  const [loading, setLoading] = useState(true)
  const [jobId, setJobId] = useState<string | null>(getActiveJobId())

  useEffect(() => {
    if (!jobId) {
      setLoading(false)
      return
    }

    const load = async () => {
      try {
        const data = await fetchJob(jobId)
        setCatalog(data)
      } finally {
        setLoading(false)
      }
    }
    void load()
  }, [jobId])

  useEffect(() => {
    if (!jobId || !catalog || catalog.status === 'completed' || catalog.status === 'failed') {
      return
    }

    const timer = window.setInterval(async () => {
      const data = await fetchJob(jobId)
      setCatalog(data)
      if (data.status === 'completed' || data.status === 'failed') {
        window.clearInterval(timer)
      }
    }, 1200)

    return () => window.clearInterval(timer)
  }, [jobId, catalog?.status])

  const matchedItems = (catalog?.items ?? [])
    .map((item) => ({
      item,
      score: buildItemSearchScore(item, search, translateAttr),
    }))
    .filter(({ score }) => search.trim().length === 0 || score > 0)
    .sort((a, b) => b.score - a.score)

  const handleReset = async () => {
    if (jobId) {
      await resetJob(jobId)
    }
    clearActiveJobId()
    setJobId(null)
    setCatalog(null)
    setSearch('')
  }

  if (loading) {
    return <div className="catalog-empty">{t('analyzing')}</div>
  }

  if (!catalog || catalog.summary.total_items === 0) {
    return (
      <div className="catalog-empty">
        <h1>{t('catalogTitle')}</h1>
        <p>{t('catalogEmpty')}</p>
        <span>{t('catalogEmptyHint')}</span>
      </div>
    )
  }

  return (
    <motion.div
      className="catalog-page"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
    >
      <div className="catalog-header">
        <div>
          <h1>{t('catalogTitle')}</h1>
          <p>{t('batchItems')}: {catalog.summary.total_items}</p>
          {catalog.status !== 'completed' && <p>{t('analyzingBatch')} {catalog.progress}%</p>}
        </div>
        <button className="btn-clear" onClick={() => void handleReset()}>
          <Trash2 size={18} />
          {t('catalogReset')}
        </button>
      </div>

      <div className="catalog-search">
        <Search size={18} />
        <input
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder={t('catalogSearchPlaceholder')}
        />
      </div>

      <div className="catalog-overview">
        <section className="result-panel">
          <h3>{t('catalogTopCategories')}</h3>
          <div className="stats-list">
            {catalog.summary.categories.map(([label, count]) => (
              <div key={label} className="stats-row">
                <span>{label}</span>
                <strong>{count}</strong>
              </div>
            ))}
          </div>
        </section>
        <section className="result-panel">
          <h3>{t('catalogTopColors')}</h3>
          <div className="stats-list">
            {catalog.summary.colors.map(([label, count]) => (
              <div key={label} className="stats-row">
                <span>{label}</span>
                <strong>{count}</strong>
              </div>
            ))}
          </div>
        </section>
        <section className="result-panel">
          <h3>{t('catalogTopPatterns')}</h3>
          <div className="stats-list">
            {catalog.summary.patterns.map(([label, count]) => (
              <div key={label} className="stats-row">
                <span>{label}</span>
                <strong>{count}</strong>
              </div>
            ))}
          </div>
        </section>
      </div>

      <section className="result-panel">
        <h3>{t('catalogTopParams')}</h3>
        <div className="stats-list">
          {catalog.summary.top_params.map(([label, count]) => (
            <div key={label} className="stats-row">
              <span>{label}</span>
              <strong>{count}</strong>
            </div>
          ))}
        </div>
      </section>

      <section className="catalog-results">
        <h2>{t('catalogMatches')}</h2>
        <div className="catalog-results-grid">
          {matchedItems.slice(0, 24).map(({ item, score }) => (
            <article key={item.id} className="catalog-item-card">
              <img src={`data:image/jpeg;base64,${item.thumbnail}`} alt={item.name_ua} />
              <div className="catalog-item-body">
                <div className="catalog-item-head">
                  <h3>{item.name_ua}</h3>
                  {search.trim().length > 0 && (
                    <span className="match-badge">{t('matchPercent')}: {score}%</span>
                  )}
                </div>
                <p>{item.summary}</p>
                <div className="tag-cloud">
                  {item.params.slice(0, 6).map((param) => (
                    <span key={`${item.id}-${param.name}`} className="mini-tag">
                      {param.name}: {param.value}
                    </span>
                  ))}
                </div>
                <div className="tag-cloud">
                  {item.tags.slice(0, 5).map((tag) => (
                    <span key={`${item.id}-${tag.label}`} className="mini-tag">
                      {translateAttr(tag.label)} {(tag.confidence * 100).toFixed(0)}%
                    </span>
                  ))}
                </div>
              </div>
            </article>
          ))}
        </div>
      </section>
    </motion.div>
  )
}
