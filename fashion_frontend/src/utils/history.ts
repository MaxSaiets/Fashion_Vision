import type { HistoryItem } from '../api'

const STORAGE_KEY = 'fashion_vision_history'
const MAX_ITEMS = 20
const MAX_IMAGE_CHARS = 120_000
const MAX_CROP_CHARS = 24_000

function compactItem(item: HistoryItem): HistoryItem {
  return {
    ...item,
    image: item.image.length > MAX_IMAGE_CHARS ? item.image.slice(0, MAX_IMAGE_CHARS) : item.image,
    tags: item.tags.slice(0, 20),
    detections: item.detections.slice(0, 12).map((d) => ({
      ...d,
      crop_base64: d.crop_base64.length > MAX_CROP_CHARS
        ? d.crop_base64.slice(0, MAX_CROP_CHARS)
        : d.crop_base64,
      attributes: d.attributes?.slice(0, 6),
    })),
  }
}

function persistHistory(items: HistoryItem[]): void {
  let next = items.slice(0, MAX_ITEMS)
  while (next.length > 0) {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(next))
      return
    } catch (error) {
      if (!(error instanceof DOMException)) throw error
      next = next.slice(0, -1)
    }
  }
  localStorage.removeItem(STORAGE_KEY)
}

export function getHistory(): HistoryItem[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (!raw) return []
    const data = JSON.parse(raw)
    return Array.isArray(data) ? data : []
  } catch {
    return []
  }
}

export function saveToHistory(item: HistoryItem): void {
  const history = getHistory()
  const updated = [compactItem(item), ...history.filter((h) => h.id !== item.id)].slice(0, MAX_ITEMS)
  persistHistory(updated)
}

export function removeFromHistory(id: string): void {
  const history = getHistory().filter((h) => h.id !== id)
  persistHistory(history)
}

export function clearHistory(): void {
  localStorage.removeItem(STORAGE_KEY)
}
