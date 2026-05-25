import { useState } from 'react'
import { motion } from 'framer-motion'
import { useLanguage } from '../i18n/LanguageContext'
import type { Detection } from '../api'

interface Props {
  imageBase64: string
  detections: Detection[]
  colors: string[]
}

export default function ImageWithOverlay({ imageBase64, detections, colors }: Props) {
  const { translateAttr } = useLanguage()
  const [hovered, setHovered] = useState<number | null>(null)
  const [imgSize, setImgSize] = useState({ w: 1, h: 1 })

  const labelWithAttributes = (d: Detection) => {
    const attrs = d.attributes?.slice(0, 2).map((attr) => translateAttr(attr.label)) ?? []
    const attrText = attrs.length > 0 ? ` · ${attrs.join(', ')}` : ''
    return `${translateAttr(d.label)}${attrText}`
  }

  return (
    <div className="image-overlay-container">
      <div className="image-wrapper">
        <img
          src={`data:image/jpeg;base64,${imageBase64}`}
          alt="Аналізоване зображення"
          className="main-image"
          onLoad={(e) => {
            const img = e.currentTarget
            setImgSize({ w: img.naturalWidth, h: img.naturalHeight })
          }}
        />
        <div className="boxes-overlay">
          {detections.map((d, i) => {
            const [xmin, ymin, xmax, ymax] = d.bbox
            const w = imgSize.w || 1
            const h = imgSize.h || 1
            const left = (xmin / w) * 100
            const top = (ymin / h) * 100
            const width = ((xmax - xmin) / w) * 100
            const height = ((ymax - ymin) / h) * 100
            const isHover = hovered === i
            return (
              <motion.div
                key={i}
                className="bbox-box"
                style={{
                  left: `${left}%`,
                  top: `${top}%`,
                  width: `${width}%`,
                  height: `${height}%`,
                  borderColor: colors[i % colors.length],
                  borderWidth: isHover ? 3 : 2,
                }}
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                onMouseEnter={() => setHovered(i)}
                onMouseLeave={() => setHovered(null)}
              >
                {isHover && (
                  <span className="bbox-label" style={{ background: colors[i % colors.length] }}>
                    {labelWithAttributes(d)} ({(d.confidence * 100).toFixed(0)}%)
                  </span>
                )}
              </motion.div>
            )
          })}
        </div>
      </div>
      <div className="overlay-legend">
        {detections.map((d, i) => (
          <div
            key={i}
            className="legend-item"
            onMouseEnter={() => setHovered(i)}
            onMouseLeave={() => setHovered(null)}
            style={{ borderColor: colors[i % colors.length] }}
          >
            <span className="legend-color" style={{ background: colors[i % colors.length] }} />
            <span>{labelWithAttributes(d)}</span>
            <span className="legend-conf">{(d.confidence * 100).toFixed(0)}%</span>
          </div>
        ))}
      </div>
    </div>
  )
}
