// Task 2 of plan 2026-11-04-lorren-distribution-step1-release: one release laid out in the feed's master folder.
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { existsSync, mkdtempSync, readFileSync, readdirSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { HEADERS, INDEX_HTML, INSTALLER, KEEP, STABLE_COPY, channelState, layoutRelease, parseNotes } from '../feed.mjs'

const FEED = 'http://127.0.0.1:8765/' // written with a trailing slash (Review Focus 5)
const NOW = new Date(2026, 10, 20, 12) // local noon on 20 November 2026: the same date wherever the test runs

// A master folder whose name has a space in it (Review Focus 2), and a fake installer of each version.
function fresh() {
  const root = mkdtempSync(join(tmpdir(), 'lorren feed '))
  return { root, site: join(root, 'site') }
}
function installer(root, version) {
  const path = join(root, `build-${version}.exe`)
  writeFileSync(path, `MZ fake ${version}`)
  return path
}
const release = (s, version, more = {}) =>
  layoutRelease({
    site: s.site, channel: 'stable', version, notes: ['First', 'Second'], installer: installer(s.root, version),
    signature: `sig-${version}`, feedUrl: FEED, now: NOW, ...more,
  })
const json = (path) => JSON.parse(readFileSync(path, 'utf8'))

// Every file under `dir`, path -> bytes: what a refused release must leave exactly as it was.
function snapshot(dir, prefix = '') {
  const out = {}
  if (!existsSync(dir)) return out
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const path = join(dir, entry.name)
    if (entry.isDirectory()) Object.assign(out, snapshot(path, `${prefix}${entry.name}/`))
    else out[`${prefix}${entry.name}`] = readFileSync(path).toString('base64')
  }
  return out
}

test('notes are read as people write them (Review Focus 4)', () => {
  assert.deepEqual(parseNotes('- First\r\n\r\n* Second  \n   \n'), ['First', 'Second'])
  assert.deepEqual(parseNotes('\n \n'), [])
  assert.deepEqual(parseNotes('-\n* \n- \r\n'), []) // bullets with nothing after them are no notes
  assert.deepEqual(parseNotes('-x and -y'), ['-x and -y']) // a dash that is no bullet stays
})

test('a stable release is laid out exactly', () => {
  const s = fresh()
  release(s, '0.1.0', { signature: 'c2lnLTAuMS4w' })
  const ch = join(s.site, 'app', 'stable')
  assert.deepEqual(json(join(ch, 'latest.json')), {
    version: '0.1.0',
    notes: 'First\nSecond',
    pub_date: NOW.toISOString(),
    platforms: { 'windows-x86_64': { signature: 'c2lnLTAuMS4w', url: 'http://127.0.0.1:8765/app/stable/Lorren_0.1.0_x64-setup.exe' } },
  })
  assert.deepEqual(json(join(ch, 'releases.json')), [{ version: '0.1.0', date: '2026-11-20', notes: ['First', 'Second'] }])
  assert.equal(readFileSync(join(ch, 'Lorren_0.1.0_x64-setup.exe.sig'), 'utf8'), 'c2lnLTAuMS4w')
  assert.equal(readFileSync(join(ch, INSTALLER('0.1.0')), 'latin1'), 'MZ fake 0.1.0')
  assert.equal(readFileSync(join(ch, STABLE_COPY), 'latin1'), 'MZ fake 0.1.0')
  assert.equal(readFileSync(join(s.site, '_headers'), 'utf8'), HEADERS)
  assert.equal(readFileSync(join(s.site, 'index.html'), 'utf8'), INDEX_HTML)
})

test('the third release prunes the first installer and keeps every release note', () => {
  const s = fresh()
  release(s, '0.1.0')
  release(s, '0.1.1')
  const third = release(s, '0.1.2')
  const ch = join(s.site, 'app', 'stable')
  assert.equal(KEEP, 2)
  assert.deepEqual(readdirSync(ch).sort(), [
    'Lorren-setup.exe', 'Lorren_0.1.1_x64-setup.exe', 'Lorren_0.1.1_x64-setup.exe.sig', 'Lorren_0.1.2_x64-setup.exe',
    'Lorren_0.1.2_x64-setup.exe.sig', 'latest.json', 'releases.json',
  ])
  assert.deepEqual(json(join(ch, 'releases.json')).map((r) => r.version), ['0.1.2', '0.1.1', '0.1.0'])
  assert.equal(json(join(ch, 'latest.json')).version, '0.1.2')
  assert.equal(readFileSync(join(ch, STABLE_COPY), 'latin1'), 'MZ fake 0.1.2')
  assert.deepEqual(third.pruned, ['app/stable/Lorren_0.1.0_x64-setup.exe', 'app/stable/Lorren_0.1.0_x64-setup.exe.sig'])
})

test('pruning goes by version order, never by name: 0.11.0 and 0.10.0 are newer than 0.9.0', () => {
  const s = fresh()
  for (const v of ['0.9.0', '0.10.0', '0.11.0']) release(s, v)
  const kept = readdirSync(join(s.site, 'app', 'stable')).filter((f) => f.endsWith('_x64-setup.exe')).sort()
  assert.deepEqual(kept, ['Lorren_0.10.0_x64-setup.exe', 'Lorren_0.11.0_x64-setup.exe'])
})

test("the date is the release's local date, not UTC's", (t) => {
  // Just after local midnight east of UTC (as in India), or just before it west of UTC, the UTC date is another day.
  const offset = new Date(2026, 10, 20, 12).getTimezoneOffset()
  if (offset === 0) {
    t.skip('this machine keeps UTC: the two dates cannot differ here')
    return
  }
  const now = offset < 0 ? new Date(2026, 10, 20, 0, 30) : new Date(2026, 10, 20, 23, 30)
  const s = fresh()
  release(s, '0.1.0', { now })
  assert.equal(json(join(s.site, 'app', 'stable', 'releases.json'))[0].date, '2026-11-20')
})

test('the same version twice is refused and changes nothing (Review Focus 1)', () => {
  const s = fresh()
  release(s, '0.1.0')
  const before = snapshot(s.site)
  assert.throws(() => release(s, '0.1.0'), { message: '0.1.0 is already in the stable feed' })
  assert.deepEqual(snapshot(s.site), before)
})

test('an older version is refused', () => {
  const s = fresh()
  release(s, '0.1.2')
  assert.throws(() => release(s, '0.1.1'), { message: "0.1.1 is not newer than 0.1.2, the stable feed's newest" })
})

test('a beta release has no stable copy and leaves stable alone', () => {
  const s = fresh()
  release(s, '0.1.0-beta.1', { channel: 'beta' })
  assert.equal(existsSync(join(s.site, 'app', 'stable')), false)
  assert.equal(existsSync(join(s.site, 'app', 'beta', STABLE_COPY)), false)
  assert.equal(
    json(join(s.site, 'app', 'beta', 'latest.json')).platforms['windows-x86_64'].url,
    'http://127.0.0.1:8765/app/beta/Lorren_0.1.0-beta.1_x64-setup.exe',
  )
})

test('no notes, a version that is none and a channel that is none are refused before anything is written', () => {
  const s = fresh()
  assert.throws(() => release(s, '0.1.0', { notes: [] }), { message: 'no release notes' })
  assert.throws(() => release(s, 'v0.1'), { message: 'v0.1 is not a version' })
  assert.throws(() => release(s, '0.1.0', { channel: 'nightly' }), { message: 'nightly is not a channel' })
  assert.equal(existsSync(s.site), false)
})

test('channelState reads a channel, and one that has never had a release', () => {
  const s = fresh()
  assert.deepEqual(channelState(s.site, 'stable'), { latest: null, releases: [] })
  release(s, '0.1.0')
  const state = channelState(s.site, 'stable')
  assert.equal(state.latest.version, '0.1.0')
  assert.deepEqual(state.releases.map((r) => r.version), ['0.1.0'])
})
