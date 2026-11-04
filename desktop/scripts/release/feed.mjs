/* ============================================================================
   FEED -- one release laid out in the update feed's master folder (plan 2026-11-04 step 1, Task 2; spec 2026-11-04 "How the
   download works"). The folder `site/` is what is deployed, whole, to the feed's Pages project:

     app/<channel>/Lorren_<version>_x64-setup.exe and .sig   the versioned installer and its signature (what the updater takes)
     app/<channel>/latest.json                               the updater's manifest (the M1 test kit's shape, which it accepted)
     app/<channel>/releases.json                             [{ version, date, notes }], newest first: the website's Updates page
     app/stable/Lorren-setup.exe                             a byte copy of the newest stable installer: the website's fixed link
     _headers, index.html                                    the Pages rules, and one line for whoever opens the address itself

   Each channel keeps the KEEP newest installers (an update that started downloading just before a release still finishes);
   releases.json keeps every release. Every refusal comes before anything is written.
   ========================================================================== */
import { copyFileSync, existsSync, mkdirSync, readFileSync, readdirSync, rmSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'
import { compareVersions, isVersion } from './semver.mjs'

export const CHANNELS = ['stable', 'beta']
export const INSTALLER = (version) => `Lorren_${version}_x64-setup.exe`
export const STABLE_COPY = 'Lorren-setup.exe'
export const KEEP = 2
// Pages already revalidates every file (max-age=0, must-revalidate), so only the website's file needs a rule: it downloads as itself.
export const HEADERS = '/app/stable/Lorren-setup.exe\n  Content-Disposition: attachment; filename="Lorren-setup.exe"\n'
export const INDEX_HTML =
  '<!doctype html><meta charset="utf-8"><title>Lorren updates</title><p>This is Lorren\'s update feed: there is nothing to read here. ' +
  'Lorren comes from the castagents website.</p>\n'

const VERSIONED = /^Lorren_(.+)_x64-setup\.exe$/

/** Release notes as people write them: one per line, without "- " or "* " bullets, blank lines and spaces dropped. */
export const parseNotes = (text) =>
  String(text)
    .split(/\r?\n/)
    .map((line) => line.trim().replace(/^[-*]\s+/, '').trim())
    .filter((line) => line !== '')

const readJson = (path, otherwise) => (existsSync(path) ? JSON.parse(readFileSync(path, 'utf8')) : otherwise)
const writeJson = (path, value) => writeFileSync(path, `${JSON.stringify(value, null, 2)}\n`)
const localDate = (d) => `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`

/** A channel's manifest (null before its first release) and its release notes, newest first. */
export function channelState(site, channel) {
  const dir = join(site, 'app', channel)
  return { latest: readJson(join(dir, 'latest.json'), null), releases: readJson(join(dir, 'releases.json'), []) }
}

/**
 * Lays one release out: the versioned pair, latest.json, releases.json, the stable copy (stable only), the pruning, _headers and
 * index.html. `installer` is the built file's path, `signature` its .sig text (Tauri's base64), `now` the release's moment.
 * Returns the files written and the files pruned, as paths under `site/` with "/" between their parts.
 */
export function layoutRelease({ site, channel, version, notes, installer, signature, feedUrl, now }) {
  if (!CHANNELS.includes(channel)) throw new Error(`${channel} is not a channel`)
  if (!notes || notes.length === 0) throw new Error('no release notes')
  if (!isVersion(version)) throw new Error(`${version} is not a version`)
  const { latest, releases } = channelState(site, channel)
  if (latest?.version === version || releases.some((r) => r.version === version)) {
    throw new Error(`${version} is already in the ${channel} feed`)
  }
  if (latest && compareVersions(version, latest.version) <= 0) {
    throw new Error(`${version} is not newer than ${latest.version}, the ${channel} feed's newest`)
  }

  const dir = join(site, 'app', channel)
  const name = INSTALLER(version)
  const at = (file) => `app/${channel}/${file}`
  mkdirSync(dir, { recursive: true })
  copyFileSync(installer, join(dir, name))
  writeFileSync(join(dir, `${name}.sig`), signature)
  const url = `${feedUrl.replace(/\/+$/, '')}/${at(name)}`
  writeJson(join(dir, 'latest.json'), {
    version,
    notes: notes.join('\n'),
    pub_date: now.toISOString(),
    platforms: { 'windows-x86_64': { signature, url } },
  })
  writeJson(join(dir, 'releases.json'), [{ version, date: localDate(now), notes: [...notes] }, ...releases])
  const written = [name, `${name}.sig`, 'latest.json', 'releases.json']
  if (channel === 'stable') {
    copyFileSync(installer, join(dir, STABLE_COPY))
    written.push(STABLE_COPY)
  }

  const pruned = []
  const kept = readdirSync(dir)
    .map((file) => VERSIONED.exec(file)?.[1])
    .filter((v) => v && isVersion(v))
    .sort((a, b) => compareVersions(b, a))
  for (const old of kept.slice(KEEP)) {
    for (const file of [INSTALLER(old), `${INSTALLER(old)}.sig`]) {
      if (!existsSync(join(dir, file))) continue
      rmSync(join(dir, file))
      pruned.push(at(file))
    }
  }

  writeFileSync(join(site, '_headers'), HEADERS)
  writeFileSync(join(site, 'index.html'), INDEX_HTML)
  return { written: [...written.map(at), '_headers', 'index.html'], pruned }
}
