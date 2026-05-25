const API_BASE = '/api'

export interface Tag {
  label: string
  confidence: number
}

export interface Detection {
  label: string
  confidence: number
  bbox: [number, number, number, number]
  crop_base64: string
  attributes?: Tag[]
}

export interface ModelInfo {
  classification?: string | null
  detection?: string | null
}

export interface AnalysisResult {
  image: string
  width?: number
  height?: number
  thumbnail?: string
  tags: Tag[]
  detections: Detection[]
  summary: string
  models?: ModelInfo
}

export interface HistoryItem extends AnalysisResult {
  id: string
  timestamp: number
  filename?: string
}

export interface FeedOptions {
  vendor: string
  category_id: string
  category_name: string
  currency: string
  price?: number | null
  old_price?: number | null
  stock_quantity?: number | null
  rozetka_category_rz_id?: string | null
  product_url_base?: string | null
  image_url_base?: string | null
  shop_name?: string
  shop_company?: string
  shop_url?: string
  draft_mode?: boolean
}

export interface OfferParam {
  name: string
  value: string
  source?: string | null
}

export interface CatalogItem {
  id: string
  article: string
  variant_group: string
  source_name: string
  vendor: string
  vendor_code?: string
  category_id: string
  category_name: string
  currency: string
  price: number
  old_price?: number | null
  stock_quantity: number
  available: boolean
  in_stock?: boolean
  group_id?: string
  primary_label?: string | null
  name: string
  name_ua: string
  description: string
  description_ua: string
  summary: string
  tags: Tag[]
  detections: Detection[]
  params: OfferParam[]
  thumbnail: string
  image_url: string
  pictures?: string[]
  product_url: string
  scores: Record<string, number>
}

export interface CatalogSummary {
  total_items: number
  categories: [string, number][]
  colors: [string, number][]
  patterns: [string, number][]
  top_params: [string, number][]
}

export interface CatalogState {
  job_id: string
  status: 'queued' | 'running' | 'completed' | 'failed'
  progress: number
  file_names: string[]
  items: CatalogItem[]
  summary: CatalogSummary
  feed_options: FeedOptions | null
  artifacts: Record<string, string | undefined>
  feeds: Record<string, string | { filename: string; item_count: number }>
  error?: string | null
}

export type BatchResponse = CatalogState

export interface BatchFormInput {
  files: File[]
  vendor?: string
  categoryId?: string
  categoryName?: string
  currency?: string
  price?: string
  oldPrice?: string
  stockQuantity?: string
  rozetkaCategoryRzId?: string
  productUrlBase?: string
  imageUrlBase?: string
  shopName?: string
  shopCompany?: string
  shopUrl?: string
  draftMode?: boolean
}

export interface EnrichFeedInput extends BatchFormInput {
  feedFile: File
}

async function handleJson<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail || 'Request failed')
  }
  return res.json()
}

export async function analyzeImage(file: File): Promise<AnalysisResult> {
  const formData = new FormData()
  formData.append('file', file)

  const res = await fetch(`${API_BASE}/analyze`, {
    method: 'POST',
    body: formData,
  })

  return handleJson<AnalysisResult>(res)
}

export async function analyzeBatch(input: BatchFormInput): Promise<BatchResponse> {
  const formData = new FormData()
  input.files.forEach((file) => formData.append('files', file))
  if (input.vendor) formData.append('vendor', input.vendor)
  if (input.categoryId) formData.append('category_id', input.categoryId)
  if (input.categoryName) formData.append('category_name', input.categoryName)
  if (input.currency) formData.append('currency', input.currency)
  if (input.price) formData.append('price', input.price)
  if (input.oldPrice) formData.append('old_price', input.oldPrice)
  if (input.stockQuantity) formData.append('stock_quantity', input.stockQuantity)
  if (input.rozetkaCategoryRzId) formData.append('rozetka_category_rz_id', input.rozetkaCategoryRzId)
  if (input.productUrlBase) formData.append('product_url_base', input.productUrlBase)
  if (input.imageUrlBase) formData.append('image_url_base', input.imageUrlBase)
  if (input.shopName) formData.append('shop_name', input.shopName)
  if (input.shopCompany) formData.append('shop_company', input.shopCompany)
  if (input.shopUrl) formData.append('shop_url', input.shopUrl)
  formData.append('draft_mode', String(input.draftMode ?? true))

  const res = await fetch(`${API_BASE}/analyze-batch`, {
    method: 'POST',
    body: formData,
  })

  return handleJson<BatchResponse>(res)
}

export async function enrichFeed(input: EnrichFeedInput): Promise<BatchResponse> {
  const formData = new FormData()
  formData.append('feed_file', input.feedFile)
  input.files.forEach((file) => formData.append('files', file))
  if (input.vendor) formData.append('vendor', input.vendor)
  if (input.categoryId) formData.append('category_id', input.categoryId)
  if (input.categoryName) formData.append('category_name', input.categoryName)
  if (input.currency) formData.append('currency', input.currency)
  if (input.price) formData.append('price', input.price)
  if (input.oldPrice) formData.append('old_price', input.oldPrice)
  if (input.stockQuantity) formData.append('stock_quantity', input.stockQuantity)
  if (input.rozetkaCategoryRzId) formData.append('rozetka_category_rz_id', input.rozetkaCategoryRzId)
  if (input.productUrlBase) formData.append('product_url_base', input.productUrlBase)
  if (input.imageUrlBase) formData.append('image_url_base', input.imageUrlBase)
  if (input.shopName) formData.append('shop_name', input.shopName)
  if (input.shopCompany) formData.append('shop_company', input.shopCompany)
  if (input.shopUrl) formData.append('shop_url', input.shopUrl)
  formData.append('draft_mode', String(input.draftMode ?? true))

  const res = await fetch(`${API_BASE}/enrich-feed`, {
    method: 'POST',
    body: formData,
  })

  return handleJson<BatchResponse>(res)
}

export async function fetchJob(jobId: string): Promise<CatalogState> {
  const res = await fetch(`${API_BASE}/jobs/${jobId}`)
  return handleJson<CatalogState>(res)
}

export async function resetJob(jobId: string): Promise<void> {
  const res = await fetch(`${API_BASE}/jobs/${jobId}/reset`, { method: 'POST' })
  await handleJson<{ status: string }>(res)
}

export function getFeedDownloadUrl(jobId: string, marketplace: 'rozetka' | 'kasta'): string {
  return `${API_BASE}/jobs/${jobId}/feed/${marketplace}`
}

export async function checkHealth(): Promise<{
  status: string
  classification: boolean
  detection: boolean
  device?: string
  models?: ModelInfo
}> {
  const res = await fetch(`${API_BASE}/health`)
  if (!res.ok) {
    throw new Error('Backend недоступний')
  }
  return res.json()
}
