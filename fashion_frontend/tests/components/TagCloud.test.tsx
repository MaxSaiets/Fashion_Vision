import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import TagCloud from '../../src/components/TagCloud'
import { LanguageProvider } from '../../src/i18n/LanguageContext'

describe('TagCloud', () => {
  it('shows empty message when no tags', () => {
    render(
      <LanguageProvider>
        <TagCloud tags={[]} />
      </LanguageProvider>
    )
    expect(screen.getByText('Теги не виявлено')).toBeInTheDocument()
  })

  it('renders tags with labels and confidence', () => {
    const tags = [
      { label: 'shirt', confidence: 0.9 },
      { label: 'pants', confidence: 0.7 },
    ]
    render(
      <LanguageProvider>
        <TagCloud tags={tags} />
      </LanguageProvider>
    )
    expect(screen.getByText('shirt')).toBeInTheDocument()
    expect(screen.getByText('штани')).toBeInTheDocument()
    expect(screen.getByText('90%')).toBeInTheDocument()
    expect(screen.getByText('70%')).toBeInTheDocument()
  })
})
