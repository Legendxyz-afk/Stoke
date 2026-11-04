// Task 3 of plan 2026-11-04-lorren-distribution-step1-release: the whole-feed check, and the deploy manifest.
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { createHash } from 'node:crypto'
import { copyFileSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { checkFeed, dropped, manifest, readDeployed, writeDeployed } from '../check.mjs'
import { INSTALLER, STABLE_COPY, layoutRelease } from '../feed.mjs'
import { makeKey, publicKeyText, signatureText, tauriForm } from './keys.mjs'

const FEED = 'http://127.0.0.1:8765'
const key = makeKey()
const PUB = tauriForm(publicKeyText(key))

// 0.1.0 then 0.1.1 on stable, each signed for real with the test key, in a master folder whose name has a space.
function goodFeed() {
  const root = mkdtempSync(join(tmpdir(), 'lorren feed '))
  const site = join(root, 'site')
  for (const version of ['0.1.0', '0.1.1']) {
    const exe = join(root, `build-${version}.exe`)
    writeFileSync(exe, `MZ fake ${version}`)
    layoutRelease({
      site, channel: 'stable', version, notes: ['A note'], installer: exe,
      signature: tauriForm(signatureText(readFileSync(exe), key)), feedUrl: FEED, now: new Date(2026, 10, 20, 12),
    })
  }
  return { root, site, ch: join(site, 'app', 'stable') }
}
const check = (feed) => checkFeed({ site: feed.site, feedUrl: FEED, appPublicKey: PUB })
function editJson(path, edit) {
  const value = JSON.parse(readFileSync(path, 'utf8'))
  edit(value)
  writeFileSync(path, JSON.stringify(value))
}

test('a good feed has no problems', () => {
  assert.deepEqual(check(goodFeed()), [])
})

test('releases.json that starts at another version than latest.json', () => {
  const f = goodFeed()
  editJson(join(f.ch, 'releases.json'), (list) => list.shift())
  assert.deepEqual(check(f), ['app/stable: latest.json names 0.1.1 but releases.json starts at 0.1.0'])
})

test('the installer latest.json points at is gone', () => {
  const f = goodFeed()
  rmSync(join(f.ch, INSTALLER('0.1.1')))
  assert.deepEqual(check(f), ['app/stable: the installer latest.json points at is not in the feed: Lorren_0.1.1_x64-setup.exe'])
})

test("latest.json's signature is not the .sig file's", () => {
  const f = goodFeed()
  editJson(join(f.ch, 'latest.json'), (m) => { m.platforms['windows-x86_64'].signature = 'c29tZXRoaW5nIGVsc2U=' })
  assert.deepEqual(check(f), ["app/stable: latest.json's signature is not the .sig file's"])
})

test('a tampered installer fails its signature, and the stable copy is no longer it', () => {
  const f = goodFeed()
  const exe = join(f.ch, INSTALLER('0.1.1'))
  const bytes = readFileSync(exe)
  bytes[2] ^= 1
  writeFileSync(exe, bytes)
  assert.deepEqual(check(f), [
    'app/stable: Lorren_0.1.1_x64-setup.exe fails its signature (signature)',
    'app/stable: Lorren-setup.exe is not the newest stable installer',
  ])
})

test("the website's file is an older installer", () => {
  const f = goodFeed()
  copyFileSync(join(f.ch, INSTALLER('0.1.0')), join(f.ch, STABLE_COPY))
  assert.deepEqual(check(f), ['app/stable: Lorren-setup.exe is not the newest stable installer'])
})

test("latest.json's url points somewhere else", () => {
  const f = goodFeed()
  editJson(join(f.ch, 'latest.json'), (m) => { m.platforms['windows-x86_64'].url = 'http://elsewhere/Lorren_0.1.1_x64-setup.exe' })
  assert.deepEqual(check(f), [`app/stable: latest.json's url is not ${FEED}/app/stable/Lorren_0.1.1_x64-setup.exe`])
})

test('a truncated latest.json is reported, and nothing else about its channel', () => {
  const f = goodFeed()
  writeFileSync(join(f.ch, 'latest.json'), '{"version": "0.1')
  assert.deepEqual(check(f), ['app/stable: latest.json does not parse'])
})

test('an installer kept beyond two', () => {
  const f = goodFeed()
  // a copy of the 0.1.0 pair under another version's name: its signature still matches its bytes, so only the count is wrong
  for (const ext of ['', '.sig']) copyFileSync(join(f.ch, `${INSTALLER('0.1.0')}${ext}`), join(f.ch, `${INSTALLER('0.0.9')}${ext}`))
  assert.deepEqual(check(f), ['app/stable: keeps 3 installers, more than 2'])
})

test('an older kept installer that no longer verifies is reported too', () => {
  const f = goodFeed()
  const old = join(f.ch, INSTALLER('0.1.0'))
  const bytes = readFileSync(old)
  bytes[2] ^= 1
  writeFileSync(old, bytes)
  assert.deepEqual(check(f), ['app/stable: Lorren_0.1.0_x64-setup.exe fails its signature (signature)'])
})

test('_headers and index.html are part of every feed', () => {
  const f = goodFeed()
  rmSync(join(f.site, '_headers'))
  rmSync(join(f.site, 'index.html'))
  assert.deepEqual(check(f), ['_headers is missing', 'index.html is missing'])
})

test('the manifest is every file and its sha256, and the deploy record reads back as written', () => {
  const f = goodFeed()
  const m = manifest(f.site)
  assert.equal(Object.keys(m).length, 9) // two pairs, latest.json, releases.json, the stable copy, _headers, index.html
  assert.equal(m['app/stable/latest.json'], createHash('sha256').update(readFileSync(join(f.ch, 'latest.json'))).digest('hex'))
  const state = join(f.root, 'state')
  assert.equal(readDeployed(state), null)
  writeDeployed(state, m)
  assert.deepEqual(readDeployed(state), m)
})

test('dropped lists what the last deploy had, is gone now, and was not pruned on purpose', () => {
  const previous = { 'a.txt': '1', 'b.txt': '2', 'c.txt': '3' }
  const current = { 'a.txt': '1', 'd.txt': '4' }
  assert.deepEqual(dropped(previous, current, ['c.txt']), ['b.txt'])
  assert.deepEqual(dropped(null, current, []), [])
})
