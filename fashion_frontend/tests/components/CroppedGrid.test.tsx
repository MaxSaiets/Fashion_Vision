import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import CroppedGrid from '../../src/components/CroppedGrid'
import { LanguageProvider } from '../../src/i18n/LanguageContext'

const mockDetections = [
  {
    label: 'shirt',
    confidence: 0.95,
    bbox: [10, 10, 100, 100] as [number, number, number, number],
    crop_base64: 'dGVzdA==',
    attributes: [{ label: 'black', confidence: 0.88 }],
  },
  {
    label: 'pants',
    confidence: 0.8,
    bbox: [50, 150, 200, 300] as [number, number, number, number],
    crop_base64: 'YmFzZTY0',
  },
]

describe('CroppedGrid', () => {
  it('renders all detection items', () => {
    render(
      <LanguageProvider>
        <CroppedGrid detections={mockDetections} colors={['#fff']} />
      </LanguageProvider>
    )
    expect(screen.getByText('shirt')).toBeInTheDocument()
    expect(screen.getByText('штани')).toBeInTheDocument()
    expect(screen.getByText('black')).toBeInTheDocument()
    expect(screen.getByText('95%')).toBeInTheDocument()
    expect(screen.getByText('80%')).toBeInTheDocument()
  })

  it('renders images for each detection', () => {
    const { container } = render(
      <LanguageProvider>
        <CroppedGrid detections={mockDetections} colors={['#fff']} />
      </LanguageProvider>
    )
    const imgs = container.querySelectorAll('img')
    expect(imgs).toHaveLength(2)
  })
})
