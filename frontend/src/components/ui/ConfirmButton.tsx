import { useEffect, useState, type ComponentProps } from 'react'
import { Button } from './Button'

interface ConfirmButtonProps extends Omit<ComponentProps<typeof Button>, 'onClick'> {
  onConfirm: () => void
  /** Label shown on the second, confirming click. */
  confirmLabel?: string
}

/**
 * A button that needs two clicks: the first arms it, the second runs `onConfirm`.
 * Disarms itself after a few seconds.
 */
export function ConfirmButton({ onConfirm, confirmLabel = 'Click again to confirm', children, ...props }: ConfirmButtonProps) {
  const [armed, setArmed] = useState(false)

  useEffect(() => {
    if (!armed) return
    const timer = setTimeout(() => setArmed(false), 4000)
    return () => clearTimeout(timer)
  }, [armed])

  return (
    <Button
      {...props}
      onClick={(e) => {
        e.preventDefault()
        e.stopPropagation()
        if (armed) {
          setArmed(false)
          onConfirm()
        } else {
          setArmed(true)
        }
      }}
    >
      {armed ? confirmLabel : children}
    </Button>
  )
}
