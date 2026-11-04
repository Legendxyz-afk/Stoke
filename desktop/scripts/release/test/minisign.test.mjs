// Task 1 of plan 2026-11-04-lorren-distribution-step1-release: minisign verification, as the updater checks what the release ships.
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { existsSync, readFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { verify } from '../minisign.mjs'
import { makeKey, makeMinisign, publicKeyText, tauriForm } from './keys.mjs'

const DESKTOP = join(dirname(fileURLToPath(import.meta.url)), '..', '..', '..')
const data = Buffer.from('MZ a fake installer, a few bytes long', 'latin1')

test('a signature made over the file verifies', () => {
  const { publicKeyText: pub, signatureText: sig } = makeMinisign(data)
  assert.deepEqual(verify(data, sig, pub), { ok: true })
})

test('one flipped byte in the file fails on the signature', () => {
  const { publicKeyText: pub, signatureText: sig } = makeMinisign(data)
  const flipped = Buffer.from(data)
  flipped[3] ^= 1
  assert.deepEqual(verify(flipped, sig, pub), { ok: false, reason: 'signature' })
})

test('another key fails on the key id', () => {
  const { signatureText: sig } = makeMinisign(data)
  const other = publicKeyText(makeKey(Buffer.from('a1a2a3a4a5a6a7a8', 'hex')))
  assert.deepEqual(verify(data, sig, other), { ok: false, reason: 'key id' })
})

test('another key that happens to carry the same key id fails on the signature', () => {
  const { signatureText: sig } = makeMinisign(data)
  assert.deepEqual(verify(data, sig, publicKeyText(makeKey())), { ok: false, reason: 'signature' })
})

test('an edited trusted comment fails', () => {
  const { publicKeyText: pub, signatureText: sig } = makeMinisign(data)
  assert.deepEqual(verify(data, sig.replace('timestamp:1', 'timestamp:2'), pub), { ok: false, reason: 'trusted comment' })
})

test('a legacy Ed signature (the data unhashed) verifies', () => {
  const { publicKeyText: pub, signatureText: sig } = makeMinisign(data, { prehashed: false })
  assert.deepEqual(verify(data, sig, pub), { ok: true })
})

test('CRLF line ends and a BOM in either text still verify', () => {
  const { publicKeyText: pub, signatureText: sig } = makeMinisign(data)
  const windows = (text) => '\uFEFF' + text.replace(/\n/g, '\r\n')
  assert.deepEqual(verify(data, windows(sig), windows(pub)), { ok: true })
})

test("Tauri's base64 form of both texts verifies", () => {
  const { publicKeyText: pub, signatureText: sig } = makeMinisign(data)
  assert.deepEqual(verify(data, tauriForm(sig), tauriForm(pub)), { ok: true })
})

// The text with its second line (the algorithm, key id and signature bytes) rewritten by `edit`.
function withBody(text, edit) {
  const lines = text.split('\n')
  lines[1] = edit(Buffer.from(lines[1], 'base64')).toString('base64')
  return lines.join('\n')
}

test('an unknown algorithm is a format failure, even over a signature that would verify', () => {
  const { publicKeyText: pub, signatureText: sig } = makeMinisign(data, { prehashed: false })
  const odd = withBody(sig, (body) => {
    const copy = Buffer.from(body)
    copy.write('Xd', 0, 'latin1')
    return copy
  })
  assert.deepEqual(verify(data, odd, pub), { ok: false, reason: 'format' })
})

test('a signature of the wrong length is a format failure', () => {
  const { publicKeyText: pub, signatureText: sig } = makeMinisign(data)
  assert.deepEqual(verify(data, withBody(sig, (body) => body.subarray(0, 70)), pub), { ok: false, reason: 'format' })
})

test('text that is no key or no signature is a format failure, never a throw', () => {
  const { publicKeyText: pub, signatureText: sig } = makeMinisign(data)
  assert.deepEqual(verify(data, 'nonsense', pub), { ok: false, reason: 'format' })
  assert.deepEqual(verify(data, sig, ''), { ok: false, reason: 'format' })
  assert.deepEqual(verify(data, sig.split('\n').slice(0, 2).join('\n'), pub), { ok: false, reason: 'format' }) // no trusted comment
})

test("the M1 kit's 0.0.2 installer verifies against its test key, and not against the wrong one", (t) => {
  const exe = join(DESKTOP, 'e2e', '.work', 'builds', '0.0.2', 'Lorren_0.0.2_x64-setup.exe')
  const keys = join(DESKTOP, '.keys')
  if (!existsSync(exe) || !existsSync(join(keys, 'test.key.pub')) || !existsSync(join(keys, 'wrong.key.pub'))) {
    t.skip('the M1 test kit has not been built here')
    return
  }
  const file = readFileSync(exe)
  const sig = readFileSync(`${exe}.sig`, 'utf8')
  assert.deepEqual(verify(file, sig, readFileSync(join(keys, 'test.key.pub'), 'utf8')), { ok: true })
  assert.deepEqual(verify(file, sig, readFileSync(join(keys, 'wrong.key.pub'), 'utf8')), { ok: false, reason: 'key id' })
})
