import { describe, it, expect, vi, beforeEach } from 'vitest'
import { analyzeImage, checkHealth } from '../src/api'

describe('API', () => {
  beforeEach(() => {
    vi.restoreAllMocks()
  })

  describe('analyzeImage', () => {
    it('sends FormData with file', async () => {
      const fetchSpy = vi.spyOn(globalThis, 'fetch').mockResolvedValue(
        new Response(
          JSON.stringify({
            image: 'base64...',
            tags: [],
            detections: [],
            summary: 'Test',
          }),
          { status: 200 }
        )
      )

      const file = new File(['content'], 'test.jpg', { type: 'image/jpeg' })
      await analyzeImage(file)

      expect(fetchSpy).toHaveBeenCalledWith(
        '/api/analyze',
        expect.objectContaining({
          method: 'POST',
          body: expect.any(FormData),
        })
      )
    })

    it('throws on error response', async () => {
      vi.spyOn(globalThis, 'fetch').mockResolvedValue(
        new Response(JSON.stringify({ detail: 'Invalid image' }), {
          status: 400,
        })
      )

      const file = new File(['x'], 'test.jpg', { type: 'image/jpeg' })
      await expect(analyzeImage(file)).rejects.toThrow('Invalid image')
    })
  })

  describe('checkHealth', () => {
    it('returns health data on success', async () => {
      vi.spyOn(globalThis, 'fetch').mockResolvedValue(
        new Response(
          JSON.stringify({
            status: 'ok',
            classification: true,
            detection: false,
          }),
          { status: 200 }
        )
      )

      const result = await checkHealth()
      expect(result).toEqual({
        status: 'ok',
        classification: true,
        detection: false,
      })
    })

    it('throws when backend unavailable', async () => {
      vi.spyOn(globalThis, 'fetch').mockResolvedValue(
        new Response('', { status: 500 })
      )

      await expect(checkHealth()).rejects.toThrow('Backend недоступний')
    })
  })
})
