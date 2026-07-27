import { Color } from 'three'

export function skuColor(index: number) {
  const hue = ((index * 137.508 + 18) % 360) / 360
  return `#${new Color().setHSL(hue, 0.68, 0.5).getHexString()}`
}
