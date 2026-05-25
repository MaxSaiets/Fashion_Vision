import { useEffect, useState, type ChangeEvent, type InputHTMLAttributes } from 'react'
import { motion } from 'framer-motion'
import { FolderOpen, Loader2, Upload } from 'lucide-react'
import {
  analyzeBatch,
  enrichFeed,
  fetchJob,
  getFeedDownloadUrl,
  type BatchFormInput,
  type BatchResponse,
} from '../api'
import { useLanguage } from '../i18n/LanguageContext'
import { setActiveJobId } from '../utils/jobs'

type DirectoryInputProps = InputHTMLAttributes<HTMLInputElement> & {
  webkitdirectory?: string
  directory?: string
}

export default function BatchPage() {
  const { t } = useLanguage()
  const [batchLoading, setBatchLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [batchFiles, setBatchFiles] = useState<File[]>([])
  const [feedFile, setFeedFile] = useState<File | null>(null)
  const [batchResult, setBatchResult] = useState<BatchResponse | null>(null)
  const [activeJobId, setActiveJobIdState] = useState<string | null>(null)
  const [batchForm, setBatchForm] = useState<Omit<BatchFormInput, 'files'>>({
    vendor: '',
    categoryId: '',
    categoryName: '',
    currency: 'UAH',
    price: '',
    oldPrice: '',
    stockQuantity: '',
    rozetkaCategoryRzId: '',
    productUrlBase: '',
    imageUrlBase: '',
    shopName: '',
    shopCompany: '',
    shopUrl: '',
    draftMode: true,
  })

  const handleBatchFileInput = (e: ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files ?? []).filter((file) => file.type.startsWith('image/'))
    setBatchFiles(files)
  }

  const handleFeedFileInput = (e: ChangeEvent<HTMLInputElement>) => {
    setFeedFile(e.target.files?.[0] ?? null)
  }

  const updateBatchField = (field: keyof Omit<BatchFormInput, 'files'>, value: string) => {
    setBatchForm((prev) => ({
      ...prev,
      [field]: value,
    }))
  }

  const handleBatchRun = async () => {
    if (batchFiles.length === 0) {
      setError(t('batchNoFiles'))
      return
    }

    setBatchLoading(true)
    setError(null)
    try {
      const data = await analyzeBatch({
        files: batchFiles,
        ...batchForm,
      })
      setBatchResult(data)
      setActiveJobIdState(data.job_id)
      setActiveJobId(data.job_id)
    } catch (e) {
      setError(e instanceof Error ? e.message : t('errorAnalyze'))
    } finally {
      setBatchLoading(false)
    }
  }

  const handleFeedEnrichment = async () => {
    if (!feedFile) {
      setError(t('feedNoFile'))
      return
    }

    setBatchLoading(true)
    setError(null)
    try {
      const data = await enrichFeed({
        feedFile,
        files: batchFiles,
        ...batchForm,
      })
      setBatchResult(data)
      setActiveJobIdState(data.job_id)
      setActiveJobId(data.job_id)
    } catch (e) {
      setError(e instanceof Error ? e.message : t('errorAnalyze'))
    } finally {
      setBatchLoading(false)
    }
  }

  useEffect(() => {
    if (!activeJobId || !batchResult || batchResult.status === 'completed' || batchResult.status === 'failed') {
      return
    }

    setBatchLoading(true)
    const timer = window.setInterval(async () => {
      try {
        const next = await fetchJob(activeJobId)
        setBatchResult(next)
        if (next.status === 'completed' || next.status === 'failed') {
          window.clearInterval(timer)
          setBatchLoading(false)
        }
      } catch (e) {
        window.clearInterval(timer)
        setBatchLoading(false)
        setError(e instanceof Error ? e.message : t('errorAnalyze'))
      }
    }, 1200)

    return () => window.clearInterval(timer)
  }, [activeJobId, batchResult?.status, t])

  const folderInputProps = {
    id: 'folder-input',
    type: 'file',
    multiple: true,
    accept: 'image/*',
    onChange: handleBatchFileInput,
    style: { display: 'none' },
    webkitdirectory: '',
    directory: '',
  } as DirectoryInputProps

  return (
    <motion.div
      className="home"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
    >
      <section className="batch-section">
        <div className="batch-header">
          <h1>{t('batchTitle')}</h1>
          <p>{t('batchSubtitle')}</p>
        </div>

        <div className="batch-grid">
          <div className="result-panel">
            <h3>{t('batchSettings')}</h3>
            <p className="batch-files-count">{t('feedDraftHint')}</p>
            <div className="feed-form">
              <label>
                <span>{t('feedVendor')}</span>
                <input value={batchForm.vendor ?? ''} onChange={(e) => updateBatchField('vendor', e.target.value)} placeholder={t('feedOptionalPlaceholder')} />
              </label>
              <label>
                <span>{t('feedCategoryId')}</span>
                <input value={batchForm.categoryId ?? ''} onChange={(e) => updateBatchField('categoryId', e.target.value)} placeholder={t('feedOptionalPlaceholder')} />
              </label>
              <label>
                <span>{t('feedCategoryName')}</span>
                <input value={batchForm.categoryName ?? ''} onChange={(e) => updateBatchField('categoryName', e.target.value)} placeholder={t('feedOptionalPlaceholder')} />
              </label>
              <label>
                <span>{t('feedPrice')}</span>
                <input value={batchForm.price ?? ''} onChange={(e) => updateBatchField('price', e.target.value)} placeholder="0.00" />
              </label>
              <label>
                <span>{t('feedOldPrice')}</span>
                <input value={batchForm.oldPrice ?? ''} onChange={(e) => updateBatchField('oldPrice', e.target.value)} placeholder={t('feedOptionalPlaceholder')} />
              </label>
              <label>
                <span>{t('feedStock')}</span>
                <input value={batchForm.stockQuantity ?? ''} onChange={(e) => updateBatchField('stockQuantity', e.target.value)} placeholder="0" />
              </label>
              <label>
                <span>{t('feedCurrency')}</span>
                <input value={batchForm.currency ?? ''} onChange={(e) => updateBatchField('currency', e.target.value)} />
              </label>
              <label>
                <span>{t('feedRozetkaCategory')}</span>
                <input value={batchForm.rozetkaCategoryRzId ?? ''} onChange={(e) => updateBatchField('rozetkaCategoryRzId', e.target.value)} placeholder={t('feedOptionalPlaceholder')} />
              </label>
              <label>
                <span>{t('feedProductUrlBase')}</span>
                <input value={batchForm.productUrlBase ?? ''} onChange={(e) => updateBatchField('productUrlBase', e.target.value)} placeholder={t('feedOptionalPlaceholder')} />
              </label>
              <label>
                <span>{t('feedImageUrlBase')}</span>
                <input value={batchForm.imageUrlBase ?? ''} onChange={(e) => updateBatchField('imageUrlBase', e.target.value)} placeholder={t('feedOptionalPlaceholder')} />
              </label>
              <label>
                <span>{t('feedShopName')}</span>
                <input value={batchForm.shopName ?? ''} onChange={(e) => updateBatchField('shopName', e.target.value)} placeholder={t('feedOptionalPlaceholder')} />
              </label>
              <label>
                <span>{t('feedShopCompany')}</span>
                <input value={batchForm.shopCompany ?? ''} onChange={(e) => updateBatchField('shopCompany', e.target.value)} placeholder={t('feedOptionalPlaceholder')} />
              </label>
              <label className="full-width">
                <span>{t('feedShopUrl')}</span>
                <input value={batchForm.shopUrl ?? ''} onChange={(e) => updateBatchField('shopUrl', e.target.value)} placeholder={t('feedOptionalPlaceholder')} />
              </label>
            </div>

            <div className="batch-actions">
              <input {...folderInputProps} />
              <input
                id="feed-input"
                type="file"
                accept=".xml,text/xml,application/xml"
                onChange={handleFeedFileInput}
                style={{ display: 'none' }}
              />
              <button className="btn-secondary" onClick={() => document.getElementById('folder-input')?.click()}>
                <FolderOpen size={18} />
                {t('batchChooseFolder')}
              </button>
              <button className="btn-secondary" onClick={() => document.getElementById('feed-input')?.click()}>
                <Upload size={18} />
                {t('feedChooseFile')}
              </button>
              <button className="btn-primary" onClick={handleBatchRun} disabled={batchLoading}>
                {batchLoading ? <Loader2 className="spin" size={18} /> : <Upload size={18} />}
                {batchLoading ? t('analyzingBatch') : t('batchRun')}
              </button>
              <button className="btn-primary" onClick={handleFeedEnrichment} disabled={batchLoading || !feedFile}>
                {batchLoading ? <Loader2 className="spin" size={18} /> : <Upload size={18} />}
                {batchLoading ? t('analyzingBatch') : t('feedEnhance')}
              </button>
            </div>

            <p className="batch-files-count">
              {batchFiles.length > 0 ? `${t('batchItems')}: ${batchFiles.length}` : t('uploadFolderHint')}
            </p>
            <p className="batch-files-count">
              {feedFile ? `${t('feedLoaded')}: ${feedFile.name}` : t('feedUploadHint')}
            </p>
          </div>

          {batchResult && (
            <div className="result-panel">
              <h3>{t('batchSummary')}</h3>
              <div className="stats-grid">
                <div className="stat-card">
                  <span>{t('batchItems')}</span>
                  <strong>{batchResult.summary.total_items}</strong>
                </div>
                <div className="stat-card">
                  <span>{t('catalogTopColors')}</span>
                  <strong>{batchResult.summary.colors[0]?.[0] ?? '—'}</strong>
                </div>
                <div className="stat-card">
                  <span>{t('catalogTopPatterns')}</span>
                  <strong>{batchResult.summary.patterns[0]?.[0] ?? '—'}</strong>
                </div>
              </div>
              <div className="feed-downloads">
                <span>{t('batchFeeds')}</span>
                <div className="feed-actions">
                  <a className="btn-secondary" href={getFeedDownloadUrl(batchResult.job_id, 'rozetka')}>
                    {t('batchFeedRozetka')}
                  </a>
                  <a className="btn-secondary" href={getFeedDownloadUrl(batchResult.job_id, 'kasta')}>
                    {t('batchFeedKasta')}
                  </a>
                </div>
              </div>
              {batchResult.status !== 'completed' && (
                <p className="batch-files-count">
                  {batchResult.status === 'failed'
                    ? batchResult.error ?? t('errorAnalyze')
                    : `${t('analyzingBatch')} ${batchResult.progress}%`}
                </p>
              )}
              <div className="catalog-preview-list">
                {batchResult.items.slice(0, 6).map((item) => (
                  <article key={item.id} className="catalog-preview-card">
                    <img src={`data:image/jpeg;base64,${item.thumbnail}`} alt={item.name_ua} />
                    <div>
                      <h4>{item.name_ua}</h4>
                      <p>{item.summary}</p>
                    </div>
                  </article>
                ))}
              </div>
            </div>
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
    </motion.div>
  )
}
