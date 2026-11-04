/* ============================================================================
   SEMVER -- what a Lorren version is, and which of two is newer, as SemVer 2.0.0 says (section 11), which is how the updater
   orders them: 0.1.0-beta.1 before 0.1.0 before 0.1.1. No build metadata, no leading "v" (plan 2026-11-04 step 1, Task 2).
   ========================================================================== */

const ID = '(?:0|[1-9]\\d*|\\d*[a-zA-Z-][0-9a-zA-Z-]*)' // one pre-release identifier: a number without leading zeros, or text
const VERSION = new RegExp(`^(0|[1-9]\\d*)\\.(0|[1-9]\\d*)\\.(0|[1-9]\\d*)(?:-(${ID}(?:\\.${ID})*))?$`)

/** Whether `v` is a version: x.y.z, with an optional -pre.release. */
export const isVersion = (v) => typeof v === 'string' && VERSION.test(v)

const numeric = (id) => /^\d+$/.test(id)

/** Negative when `a` comes before `b`, positive when after, 0 when they are the same version. */
export function compareVersions(a, b) {
  const [, a1, a2, a3, apre] = VERSION.exec(a)
  const [, b1, b2, b3, bpre] = VERSION.exec(b)
  for (const [x, y] of [[a1, b1], [a2, b2], [a3, b3]]) if (Number(x) !== Number(y)) return Number(x) - Number(y)
  if (apre === undefined || bpre === undefined) return (apre === undefined) - (bpre === undefined) // a release after its pre-releases
  const as = apre.split('.')
  const bs = bpre.split('.')
  for (let i = 0; i < Math.min(as.length, bs.length); i++) {
    const [x, y] = [as[i], bs[i]]
    if (x === y) continue
    if (numeric(x) && numeric(y)) return Number(x) - Number(y)
    if (numeric(x) !== numeric(y)) return numeric(x) ? -1 : 1 // a number before text
    return x < y ? -1 : 1 // text by its characters, in ASCII order
  }
  return as.length - bs.length
}
