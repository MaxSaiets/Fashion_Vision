import { createContext, useContext, useState, useCallback, type ReactNode } from 'react'
import { ui, attrUk, type Lang } from './translations'

interface LanguageContextType {
  lang: Lang
  setLang: (l: Lang) => void
  t: (key: string) => string
  translateAttr: (label: string) => string
}

const LanguageContext = createContext<LanguageContextType | null>(null)

export function LanguageProvider({ children }: { children: ReactNode }) {
  const [lang, setLangState] = useState<Lang>('uk')

  const setLang = useCallback((l: Lang) => {
    setLangState(l)
  }, [])

  const t = useCallback(
    (key: string) => ui[lang][key] ?? key,
    [lang]
  )

  const translateAttr = useCallback(
    (label: string) => {
      if (lang === 'en') return label
      return attrUk[label] ?? label
    },
    [lang]
  )

  return (
    <LanguageContext.Provider value={{ lang, setLang, t, translateAttr }}>
      {children}
    </LanguageContext.Provider>
  )
}

export function useLanguage() {
  const ctx = useContext(LanguageContext)
  if (!ctx) throw new Error('useLanguage must be used within LanguageProvider')
  return ctx
}
