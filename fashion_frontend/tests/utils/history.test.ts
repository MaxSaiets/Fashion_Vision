import { describe, it, expect, beforeEach } from 'vitest'
import {
  getHistory,
  saveToHistory,
  removeFromHistory,
  clearHistory,
} from '../../src/utils/history'

const mockItem = {
  id: 'test-id',
  timestamp: Date.now(),
  image: 'base64',
  tags: [{ label: 'shirt', confidence: 0.9 }],
  detections: [],
  summary: 'Test summary',
}

describe('history utils', () => {
  beforeEach(() => {
    clearHistory()
  })

  it('getHistory returns empty array initially', () => {
    expect(getHistory()).toEqual([])
  })

  it('saveToHistory adds item', () => {
    saveToHistory(mockItem)
    const history = getHistory()
    expect(history).toHaveLength(1)
    expect(history[0].id).toBe('test-id')
    expect(history[0].tags[0].label).toBe('shirt')
  })

  it('saveToHistory prepends new items', () => {
    saveToHistory({ ...mockItem, id: 'first' })
    saveToHistory({ ...mockItem, id: 'second' })
    const history = getHistory()
    expect(history[0].id).toBe('second')
    expect(history[1].id).toBe('first')
  })

  it('removeFromHistory removes by id', () => {
    saveToHistory(mockItem)
    removeFromHistory('test-id')
    expect(getHistory()).toHaveLength(0)
  })

  it('clearHistory removes all', () => {
    saveToHistory(mockItem)
    saveToHistory({ ...mockItem, id: 'other' })
    clearHistory()
    expect(getHistory()).toEqual([])
  })
})
