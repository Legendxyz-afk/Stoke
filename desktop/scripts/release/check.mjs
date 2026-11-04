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

/** The feed's problems, as sentences; [] when it may be deployed. `appPublicKey` is the public key's text (either form). */
export function checkFeed({ site, feedUrl, appPublicKey }) {
  const problems = []
  for (const channel of CHANNELS) {
    if (existsSync(join(site, 'app', channel))) checkChannel(site, channel, feedUrl, appPublicKey, problems)
  }
  for (const file of ['_headers', 'index.html']) if (!existsSync(join(site, file))) problems.push(`${file} is missing`)
  if (existsSync(join(site, 'catalog')) && !readJson(join(site, 'catalog', 'catalog.json'))) {
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
