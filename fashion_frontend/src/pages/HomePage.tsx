import { useState, type ChangeEvent, type DragEvent } from 'react'
import { motion } from 'framer-motion'
import { Loader2, Upload } from 'lucide-react'
import { analyzeImage, type AnalysisResult } from '../api'
import { useLanguage } from '../i18n/LanguageContext'
import ImageWithOverlay from '../components/ImageWithOverlay'
import TagCloud from '../components/TagCloud'
import CroppedGrid from '../components/CroppedGrid'
import { saveToHistory } from '../utils/history'

const COLORS = ['#d4a574', '#7eb8da', '#9fd5a1', '#e8a87c', '#c38d9e', '#85cdca']
const PRIMARY_DEMO_ITEMS = new Set([
  'shirt, blouse',
  'top, t-shirt, sweatshirt',
  'sweater',
  'cardigan',
  'jacket',
  'vest',
  'pants',
  'shorts',
  'skirt',
  'coat',
  'dress',
  'jumpsuit',
  'cape',
])
const SECONDARY_DEMO_ITEMS = new Set([
  'shoe',
  'bag, wallet',
  'hat',
  'tie',
  'glove',
  'watch',
  'belt',
  'glasses',
  'sock',
  'tights, stockings',
  'scarf',
])

function getSummaryDetections(detections: AnalysisResult['detections']) {
  const uniqueByLabel = new Map<string, AnalysisResult['detections'][number]>()
  detections.forEach((d) => {
    const previous = uniqueByLabel.get(d.label)
    if (!previous || d.confidence > previous.confidence) {
      uniqueByLabel.set(d.label, d)
    }
  })
  const unique = [...uniqueByLabel.values()]
  const primary = unique.filter((d) => PRIMARY_DEMO_ITEMS.has(d.label))
  const secondary = unique.filter((d) => SECONDARY_DEMO_ITEMS.has(d.label))
  const fallback = unique.filter((d) => !PRIMARY_DEMO_ITEMS.has(d.label) && !SECONDARY_DEMO_ITEMS.has(d.label))
  const ordered = primary.length > 0 ? [...primary, ...secondary.slice(0, 2)] : [...secondary, ...fallback]
  return ordered.sort((a, b) => b.confidence - a.confidence).slice(0, 5)
}

export default function HomePage() {
  const { t, translateAttr } = useLanguage()
  const [, setFile] = useState<File | null>(null)
  const [result, setResult] = useState<AnalysisResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [dragActive, setDragActive] = useState(false)

  const handleAnalyze = async (f: File) => {
    setLoading(true)
    setError(null)
    setResult(null)
    setFile(f)

    try {
      const data = await analyzeImage(f)
      setResult(data)
      saveToHistory({
        ...data,
        id: crypto.randomUUID(),
        timestamp: Date.now(),
        filename: f.name,
      })
    } catch (e) {
      setError(e instanceof Error ? e.message : t('errorAnalyze'))
    } finally {
      setLoading(false)
    }
  }

  const handleDrop = (e: DragEvent) => {
    e.preventDefault()
    setDragActive(false)
    const f = e.dataTransfer.files[0]
    if (f?.type.startsWith('image/')) void handleAnalyze(f)
  }

  const handleFileInput = (e: ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0]
    if (f) void handleAnalyze(f)
  }

  return (
    <motion.div
      className="home"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
    >
      <section className="hero">
        <h1>{t('heroTitle')}</h1>
        <p className="subtitle">{t('heroSubtitle')}</p>
      </section>

      <section className="upload-section">
        <div
          className={`upload-zone ${dragActive ? 'drag-active' : ''} ${loading ? 'loading' : ''}`}
          onDragOver={(e) => { e.preventDefault(); setDragActive(true) }}
          onDragLeave={() => setDragActive(false)}
          onDrop={handleDrop}
          onClick={() => !loading && document.getElementById('file-input')?.click()}
        >
          <input
            id="file-input"
            type="file"
            accept="image/*"
            onChange={handleFileInput}
            style={{ display: 'none' }}
          />
          {loading ? (
            <>
              <Loader2 className="spin" size={48} />
              <span>{t('analyzing')}</span>
            </>
          ) : (
            <>
              <Upload size={48} />
              <span>{t('uploadHint')}</span>
              <span className="hint">{t('uploadHintFormat')}</span>
            </>
          )}
        </div>
      </section>

      {error && (
        <motion.div
          className="error-banner"
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
        >
          {error}
        </motion.div>
      )}

      {result && (
        <motion.section
          className="results"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
        >
          <h2>{t('resultsTitle')}</h2>

          {(() => {
            const parts: string[] = []
            if (result.detections.length > 0) {
              const summaryDetections = getSummaryDetections(result.detections)
              const items = summaryDetections
                .map((d) => `${translateAttr(d.label)} (${(d.confidence * 100).toFixed(0)}%)`)
              parts.push(`${t('summaryDetected')}: ${items.join(', ')}.`)

              const localized = summaryDetections
                .filter((d) => d.attributes && d.attributes.length > 0)
                .slice(0, 4)
                .map((d) => {
                  const attrs = d.attributes!.slice(0, 3).map((x) => translateAttr(x.label)).join(', ')
                  return `${translateAttr(d.label)}: ${attrs}`
                })
              if (localized.length > 0) {
                parts.push(`${t('summaryElementAttributes')}: ${localized.join('; ')}.`)
              }
            }
            if (result.tags.length > 0) {
              const topTags = result.tags.slice(0, 5).map((x) => translateAttr(x.label))
              parts.push(`${t('summaryTags')}: ${topTags.join(', ')}.`)
            }
            const summary = parts.length > 0 ? parts.join(' ') : t('summaryDone')
            return (
              <div className="summary-card">
                <p>{summary}</p>
              </div>
            )
          })()}

          <div className="results-grid">
            <div className="result-panel image-panel">
              <h3>{t('originalHighlight')}</h3>
              <ImageWithOverlay
                imageBase64={result.image}
                detections={result.detections}
                colors={COLORS}
              />
            </div>

            <div className="result-panel tags-panel">
              <h3>{t('tagsAttributes')}</h3>
              <TagCloud tags={result.tags} />
            </div>
          </div>

          {result.detections.length > 0 && (
            <div className="result-panel crops-panel">
              <h3>{t('detectedElements')}</h3>
              <CroppedGrid detections={result.detections} colors={COLORS} />
            </div>
          )}
        </motion.section>
      )}
    </motion.div>
  )
}
