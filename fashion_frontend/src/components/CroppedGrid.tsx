import { motion } from 'framer-motion'
import { useLanguage } from '../i18n/LanguageContext'
import type { Detection } from '../api'

interface Props {
  detections: Detection[]
  colors: string[]
}

export default function CroppedGrid({ detections, colors }: Props) {
  const { translateAttr } = useLanguage()

  return (
    <div className="cropped-grid">
      {detections.map((d, i) => (
        <motion.div
          key={i}
          className="cropped-item"
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: i * 0.05 }}
        >
          <div className="cropped-img-wrap" style={{ borderColor: colors[i % colors.length] }}>
            <img
              src={`data:image/jpeg;base64,${d.crop_base64}`}
              alt={translateAttr(d.label)}
            />
          </div>
          <div className="cropped-info">
            <span className="cropped-label">{translateAttr(d.label)}</span>
            <span className="cropped-conf">{(d.confidence * 100).toFixed(0)}%</span>
            {d.attributes && d.attributes.length > 0 && (
              <div className="crop-attrs">
                {d.attributes.slice(0, 4).map((attr) => (
                  <span key={attr.label} className="crop-attr">
                    {translateAttr(attr.label)}
                  </span>
                ))}
              </div>
            )}
          </div>
        </motion.div>
      ))}
    </div>
  )
}
