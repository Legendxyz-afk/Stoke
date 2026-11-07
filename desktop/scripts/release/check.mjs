/* ============================================================================
   CHECK -- the whole feed, checked before any deploy (plan 2026-11-04 step 1, Task 3; spec 2026-11-04: "a deploy replaces the
   site: both channels and the catalog must be complete and every manifest must point at a file that is there and whose
   signature verifies"), and the record of what the last deploy held, so that a deploy can never drop a file it did not mean to.

   checkFeed returns the problems as sentences, each channel's prefixed with its folder; an empty list is a feed that may be
   deployed. A channel whose latest.json is missing or is not a manifest gets that one sentence: nothing else about it can be read.
   ========================================================================== */
import { createHash } from 'node:crypto'
import { existsSync, mkdirSync, readFileSync, readdirSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'
import { CHANNELS, INSTALLER, KEEP, STABLE_COPY } from './feed.mjs'
import { verify } from './minisign.mjs'
import { isVersion } from './semver.mjs'

const VERSIONED = /^Lorren_(.+)_x64-setup\.exe$/
const DEPLOYED = 'deployed.json'

const sha256 = (path) => createHash('sha256').update(readFileSync(path)).digest('hex')
function readJson(path) {
  try {
    return { value: JSON.parse(readFileSync(path, 'utf8')) }
  } catch {
    return null
  }
}

function checkChannel(site, channel, feedUrl, publicKey, problems) {
  const dir = join(site, 'app', channel)
  const say = (text) => problems.push(`app/${channel}: ${text}`)

  if (!existsSync(join(dir, 'latest.json'))) return say('latest.json is missing')
  const latest = readJson(join(dir, 'latest.json'))?.value
  const platform = latest?.platforms?.['windows-x86_64']
  if (typeof latest?.version !== 'string' || typeof platform?.url !== 'string' || typeof platform?.signature !== 'string') {
    return say('latest.json does not parse')
  }

  const releases = readJson(join(dir, 'releases.json'))?.value
  if (!Array.isArray(releases)) say('releases.json does not parse')
  else if (releases[0]?.version !== latest.version) {
    say(`latest.json names ${latest.version} but releases.json starts at ${releases[0]?.version ?? 'nothing'}`)
  }

  const name = INSTALLER(latest.version)
  const expected = `${feedUrl.replace(/\/+$/, '')}/app/${channel}/${name}`
  if (platform.url !== expected) say(`latest.json's url is not ${expected}`)

  const installer = join(dir, name)
  if (!existsSync(installer)) say(`the installer latest.json points at is not in the feed: ${name}`)
  else {
    if (!existsSync(`${installer}.sig`)) say(`${name} has no .sig`)
    else {
      const sig = readFileSync(`${installer}.sig`, 'utf8').trim()
      if (platform.signature !== sig) say("latest.json's signature is not the .sig file's")
      const result = verify(readFileSync(installer), sig, publicKey)
      if (!result.ok) say(`${name} fails its signature (${result.reason})`)
    }
    if (channel === 'stable') {
      const copy = join(dir, STABLE_COPY)
      if (!existsSync(copy)) say(`${STABLE_COPY} is missing`)
      else if (sha256(copy) !== sha256(installer)) say(`${STABLE_COPY} is not the newest stable installer`)
    }
  }

  const versions = readdirSync(dir).map((file) => VERSIONED.exec(file)?.[1]).filter(Boolean)
  if (versions.length > KEEP) say(`keeps ${versions.length} installers, more than ${KEEP}`)
  for (const version of versions) {
    if (version === latest.version) continue // checked above
    const other = join(dir, INSTALLER(version))
    if (!existsSync(`${other}.sig`)) {
      say(`${INSTALLER(version)} has no .sig`)
      continue
    }
    const result = verify(readFileSync(other), readFileSync(`${other}.sig`, 'utf8').trim(), publicKey)
    if (!result.ok) say(`${INSTALLER(version)} fails its signature (${result.reason})`)
  }
}

// ---- The model catalog (step 2): the entry rules, the SAME list as src-tauri/src/catalog.rs `validate_entry` ----

const MAX_SIZE = 68719476736 // 64 GiB
const ID_RULE = /^[a-z0-9][a-z0-9._-]{0,63}$/
const FILE_RULE = /^[A-Za-z0-9][A-Za-z0-9._-]{0,127}\.gguf$/i
const text = (v, min, max) => typeof v === 'string' && v.length >= min && v.length <= max
const isObject = (v) => v !== null && typeof v === 'object' && !Array.isArray(v)

/** One catalog entry's problems, each starting with the field it is about; [] for a good entry. `insecure`: http:// urls are allowed (a test build). */
export function validateEntry(entry, { insecure = false } = {}) {
  if (!isObject(entry)) return ['entry is not an object']
  const out = []
  const bad = (field, rule) => out.push(`${field} ${rule}`)
  const url = (v) => typeof v === 'string' && (v.startsWith('https://') || (insecure && v.startsWith('http://'))) && v.length > 8
  if (typeof entry.id !== 'string' || !ID_RULE.test(entry.id)) bad('id', 'is not 1 to 64 lowercase letters, digits, dots, dashes or underscores, starting with a letter or digit')
  if (!text(entry.name, 1, 80)) bad('name', 'is not 1 to 80 characters')
  if (!text(entry.description, 0, 400)) bad('description', 'is not text of at most 400 characters')
  if (!Number.isSafeInteger(entry.version) || entry.version < 1) bad('version', 'is not a positive whole number')
  if (entry.architecture !== 'qwen3' && entry.architecture !== 'llama') bad('architecture', 'is not qwen3 or llama')
  if (!text(entry.quantization, 1, 16)) bad('quantization', 'is not 1 to 16 characters')
  if (typeof entry.file !== 'string' || !FILE_RULE.test(entry.file) || entry.file.includes('..')) bad('file', 'is not a plain .gguf file name')
  if (!Number.isSafeInteger(entry.size) || entry.size < 1 || entry.size > MAX_SIZE) bad('size', `is not a whole number from 1 to ${MAX_SIZE}`)
  if (typeof entry.sha256 !== 'string' || !/^[0-9a-f]{64}$/.test(entry.sha256)) bad('sha256', 'is not 64 lowercase hex digits')
  if (!url(entry.url)) bad('url', insecure ? 'is not an http:// or https:// address' : 'is not an https:// address')
  if (!isObject(entry.licence)) bad('licence', 'is missing')
  else {
    if (!text(entry.licence.name, 1, 80)) bad('licence.name', 'is not 1 to 80 characters')
    if (typeof entry.licence.url !== 'string' || !entry.licence.url.startsWith('https://') || entry.licence.url.length <= 8) bad('licence.url', 'is not an https:// address')
  }
  if (!isVersion(entry.min_app_version)) bad('min_app_version', 'is not a version')
  if (entry.system_prompt !== null && !text(entry.system_prompt, 0, 4000)) bad('system_prompt', 'is neither null nor text of at most 4000 characters')
  return out
}

/** Every problem of a catalog's `models` list: each entry's (named by its place and id), then ids and files that appear twice. */
export function validateModels(models) {
  if (!Array.isArray(models)) return ['models is not a list']
  const out = []
  models.forEach((entry, i) => {
    const label = typeof entry?.id === 'string' ? `models[${i}] (${entry.id})` : `models[${i}]`
    for (const problem of validateEntry(entry)) out.push(`${label}: ${problem}`)
  })
  // A file name collides with its other-case spelling on Windows, so files are compared in lower case.
  for (const [field, key] of [['id', (v) => v], ['file', (v) => v.toLowerCase()]]) {
    const first = new Map()
    const said = new Set()
    for (const entry of models) {
      const value = entry?.[field]
      if (typeof value !== 'string') continue
      const k = key(value)
      if (!first.has(k)) first.set(k, value)
      else if (!said.has(k)) {
        out.push(`${field} "${first.get(k)}" appears twice`)
        said.add(k)
      }
    }
  }
  return out
}

/**
 * The catalog's own problems, as sentences prefixed `catalog: `: its signature (the catalog key's), its format and its entries.
 * [] when `catalogPublicKey` is not given: a config without one neither requires nor compiles in a catalog.
 */
export function checkCatalog({ site, catalogPublicKey }) {
  if (!catalogPublicKey) return []
  const say = (sentence) => `catalog: ${sentence}`
  const path = join(site, 'catalog', 'catalog.json')
  if (!existsSync(path)) return [say('catalog/catalog.json is missing')]
  const problems = []
  const sigPath = `${path}.minisig`
  if (!existsSync(sigPath)) problems.push(say('catalog/catalog.json.minisig is missing'))
  else {
    const result = verify(readFileSync(path), readFileSync(sigPath, 'utf8'), catalogPublicKey)
    if (!result.ok) problems.push(say(`catalog/catalog.json fails its signature (${result.reason})`))
  }
  const parsed = readJson(path)?.value
  if (!isObject(parsed)) problems.push(say('catalog/catalog.json does not parse'))
  else if (parsed.format !== 1) problems.push(say('catalog/catalog.json is not format 1'))
  else for (const problem of validateModels(parsed.models)) problems.push(say(problem))
  return problems
}

/**
 * The feed's problems, as sentences; [] when it may be deployed. `appPublicKey` is the public key's text (either form);
 * `catalogPublicKey` (optional, the same) makes the catalog part of the feed: required, signed by that key, its entries checked.
 */
export function checkFeed({ site, feedUrl, appPublicKey, catalogPublicKey }) {
  const problems = []
  for (const channel of CHANNELS) {
    if (existsSync(join(site, 'app', channel))) checkChannel(site, channel, feedUrl, appPublicKey, problems)
  }
  for (const file of ['_headers', 'index.html', '404.html']) if (!existsSync(join(site, file))) problems.push(`${file} is missing`)
  if (catalogPublicKey) problems.push(...checkCatalog({ site, catalogPublicKey }))
  else if (existsSync(join(site, 'catalog')) && !readJson(join(site, 'catalog', 'catalog.json'))) {
    problems.push('catalog/catalog.json does not parse')
  }
  return problems
}

/** Every file under `site/`, as a path with "/" between its parts, to its sha256 (hex). */
export function manifest(site) {
  const out = {}
  const walk = (dir, prefix) => {
    for (const entry of readdirSync(dir, { withFileTypes: true })) {
      const path = join(dir, entry.name)
      if (entry.isDirectory()) walk(path, `${prefix}${entry.name}/`)
      else out[`${prefix}${entry.name}`] = sha256(path)
    }
  }
  if (existsSync(site)) walk(site, '')
  return out
}

/** What the last deploy held (its manifest), or null before the first deploy. */
export const readDeployed = (state) => (existsSync(join(state, DEPLOYED)) ? JSON.parse(readFileSync(join(state, DEPLOYED), 'utf8')) : null)

/** Records what a deploy held. */
export function writeDeployed(state, record) {
  mkdirSync(state, { recursive: true })
  writeFileSync(join(state, DEPLOYED), `${JSON.stringify(record, null, 2)}\n`)
}

/** The files the last deploy held that are gone now, other than those pruned on purpose: what this deploy would drop. */
export const dropped = (previous, current, pruned) =>
  previous ? Object.keys(previous).filter((path) => !(path in current) && !pruned.includes(path)) : []
