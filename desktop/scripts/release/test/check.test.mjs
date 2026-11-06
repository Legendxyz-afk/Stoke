// Task 3 of plan 2026-11-04-lorren-distribution-step1-release: the whole-feed check, and the deploy manifest.
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { createHash } from 'node:crypto'
import { copyFileSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { checkCatalog, checkFeed, dropped, manifest, readDeployed, writeDeployed } from '../check.mjs'
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

// ---- the catalog (step 2, Task 2) ----
const catalogKey = makeKey(Buffer.from('c1c2c3c4c5c6c7c8', 'hex'))
const CATALOG_PUB = tauriForm(publicKeyText(catalogKey))
const model = (id, file, change = {}) => ({
  id, name: 'A model', version: 1, description: 'A model.', architecture: 'llama', quantization: 'Q4_K_M', file, size: 1000, sha256: 'b'.repeat(64),
  url: 'https://huggingface.co/x/y/resolve/c/f.gguf', licence: { name: 'MIT', url: 'https://example.com/LICENSE' },
  source: { repo: 'x/y', commit: 'c', path: 'f.gguf' }, system_prompt: null, min_app_version: '0.1.0', ...change,
})
function withCatalog(models = [model('a', 'a.gguf'), model('b', 'b.gguf')], { signWith = catalogKey, minisig = true } = {}) {
  const f = goodFeed()
  mkdirSync(join(f.site, 'catalog'))
  const bytes = Buffer.from(`${JSON.stringify({ format: 1, generated: '2026-11-06T00:00:00.000Z', models }, null, 2)}
`)
  writeFileSync(join(f.site, 'catalog', 'catalog.json'), bytes)
  if (minisig) writeFileSync(join(f.site, 'catalog', 'catalog.json.minisig'), signatureText(bytes, signWith))
  return f
}
const checkWith = (f) => checkFeed({ site: f.site, feedUrl: FEED, appPublicKey: PUB, catalogPublicKey: CATALOG_PUB })

test('a feed with a good catalog has no problems; without a catalog key the catalog is not required', () => {
  assert.deepEqual(checkWith(withCatalog()), [])
  assert.deepEqual(check(goodFeed()), [])
  assert.deepEqual(checkCatalog({ site: withCatalog().site, catalogPublicKey: CATALOG_PUB }), [])
})

test('a feed whose catalog fails its signature, or has a bad entry, or has no .minisig, is not deployable', () => {
  assert.deepEqual(checkWith(withCatalog(undefined, { signWith: makeKey(Buffer.from('a1a2a3a4a5a6a7a8', 'hex')) })), [
    'catalog: catalog/catalog.json fails its signature (key id)',
  ])
  assert.deepEqual(checkWith(withCatalog([model('a', 'a.gguf', { size: 0 })])), [
    'catalog: models[0] (a): size is not a whole number from 1 to 68719476736',
  ])
  assert.deepEqual(checkWith(withCatalog(undefined, { minisig: false })), ['catalog: catalog/catalog.json.minisig is missing'])
  const tampered = withCatalog()
  writeFileSync(join(tampered.site, 'catalog', 'catalog.json'), readFileSync(join(tampered.site, 'catalog', 'catalog.json'), 'utf8').replace('A model.', 'A modeL.'))
  assert.deepEqual(checkWith(tampered), ['catalog: catalog/catalog.json fails its signature (signature)'])
  assert.deepEqual(checkWith(withCatalog([model('a', 'a.gguf'), model('a', 'c.gguf')])), ['catalog: id "a" appears twice'])
  assert.deepEqual(checkWith(withCatalog([model('a', 'a.gguf'), model('b', 'a.gguf')])), ['catalog: file "a.gguf" appears twice'])
})

test('with a catalog key configured, a feed without a catalog is not deployable', () => {
  assert.deepEqual(checkWith(goodFeed()), ['catalog: catalog/catalog.json is missing'])
})

test('a catalog that does not parse, or is not format 1, is reported', () => {
  const f = withCatalog()
  writeFileSync(join(f.site, 'catalog', 'catalog.json'), '{"format": 1')
  assert.deepEqual(checkWith(f), ['catalog: catalog/catalog.json fails its signature (signature)', 'catalog: catalog/catalog.json does not parse'])
  const bytes = Buffer.from(JSON.stringify({ format: 2, models: [] }))
  writeFileSync(join(f.site, 'catalog', 'catalog.json'), bytes)
  writeFileSync(join(f.site, 'catalog', 'catalog.json.minisig'), signatureText(bytes, catalogKey))
  assert.deepEqual(checkWith(f), ['catalog: catalog/catalog.json is not format 1'])
})
