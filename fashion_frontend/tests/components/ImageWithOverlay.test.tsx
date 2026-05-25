import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import ImageWithOverlay from '../../src/components/ImageWithOverlay'
import { LanguageProvider } from '../../src/i18n/LanguageContext'

const base64Image = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=='

describe('ImageWithOverlay', () => {
  it('renders image', () => {
    render(
      <LanguageProvider>
        <ImageWithOverlay
          imageBase64={base64Image}
          detections={[]}
          colors={['#fff']}
        />
      </LanguageProvider>
    )
    const img = screen.getByAltText('Аналізоване зображення')
    expect(img).toBeInTheDocument()
  })

  it('renders legend for detections', () => {
    const detections = [
      {
        label: 'shirt',
        confidence: 0.9,
        bbox: [0, 0, 50, 50] as [number, number, number, number],
        crop_base64: 'x',
        attributes: [{ label: 'black', confidence: 0.85 }],
      },
    ]
    render(
      <LanguageProvider>
        <ImageWithOverlay
          imageBase64={base64Image}
          detections={detections}
          colors={['#d4a574']}
        />
      </LanguageProvider>
    )
    expect(screen.getByText('shirt · black')).toBeInTheDocument()
    expect(screen.getByText('90%')).toBeInTheDocument()
  })
})
