import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import Layout from '../../src/components/Layout'
import { LanguageProvider } from '../../src/i18n/LanguageContext'

describe('Layout', () => {
  it('renders logo and navigation', () => {
    render(
      <MemoryRouter>
        <LanguageProvider>
          <Layout>
            <div>Content</div>
          </Layout>
        </LanguageProvider>
      </MemoryRouter>
    )
    expect(screen.getByText('Fashion Vision')).toBeInTheDocument()
    expect(screen.getByText('Аналіз')).toBeInTheDocument()
    expect(screen.getByText('Пакетний аналіз')).toBeInTheDocument()
    expect(screen.getByText('Каталог')).toBeInTheDocument()
    expect(screen.getByText('Історія')).toBeInTheDocument()
    expect(screen.getByText('Content')).toBeInTheDocument()
  })
})
