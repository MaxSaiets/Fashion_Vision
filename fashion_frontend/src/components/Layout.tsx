import { Link, useLocation } from 'react-router-dom'
import { Camera, ChartColumn, FolderKanban, History, Languages } from 'lucide-react'
import { useLanguage } from '../i18n/LanguageContext'

interface LayoutProps {
  children: React.ReactNode
}

export default function Layout({ children }: LayoutProps) {
  const location = useLocation()
  const { lang, setLang, t } = useLanguage()

  return (
    <div className="layout">
      <header className="header">
        <Link to="/" className="logo">
          <span className="logo-icon">◈</span>
          <span>{t('appTitle')}</span>
        </Link>
        <nav className="nav">
          <div className="lang-switcher">
            <Languages size={18} />
            <button
              className={`lang-btn ${lang === 'uk' ? 'active' : ''}`}
              onClick={() => setLang('uk')}
              title="Українська"
            >
              UA
            </button>
            <button
              className={`lang-btn ${lang === 'en' ? 'active' : ''}`}
              onClick={() => setLang('en')}
              title="English"
            >
              EN
            </button>
          </div>
          <Link
            to="/"
            className={`nav-link ${location.pathname === '/' ? 'active' : ''}`}
          >
            <Camera size={18} />
            {t('navAnalyze')}
          </Link>
          <Link
            to="/catalog"
            className={`nav-link ${location.pathname === '/catalog' ? 'active' : ''}`}
          >
            <ChartColumn size={18} />
            {t('navCatalog')}
          </Link>
          <Link
            to="/history"
            className={`nav-link ${location.pathname === '/history' ? 'active' : ''}`}
          >
            <History size={18} />
            {t('navHistory')}
          </Link>
          <Link
            to="/batch"
            className={`nav-link ${location.pathname === '/batch' ? 'active' : ''}`}
          >
            <FolderKanban size={18} />
            {t('navBatch')}
          </Link>
        </nav>
      </header>

      <main className="main">
        {children}
      </main>

      <footer className="footer">
        <span>{t('footer')}</span>
        <span>•</span>
        <span>Fashionpedia</span>
      </footer>
    </div>
  )
}
