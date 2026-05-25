import { Routes, Route } from 'react-router-dom'
import { AnimatePresence } from 'framer-motion'
import { LanguageProvider } from './i18n/LanguageContext'
import Layout from './components/Layout'
import './App.css'
import HomePage from './pages/HomePage'
import HistoryPage from './pages/HistoryPage'
import CatalogPage from './pages/CatalogPage'
import BatchPage from './pages/BatchPage'

function App() {
  return (
    <LanguageProvider>
    <Layout>
      <AnimatePresence mode="wait">
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/batch" element={<BatchPage />} />
          <Route path="/catalog" element={<CatalogPage />} />
          <Route path="/history" element={<HistoryPage />} />
        </Routes>
      </AnimatePresence>
    </Layout>
    </LanguageProvider>
  )
}

export default App
