// Test keys and signatures in minisign's own format, made here with Node's Ed25519, so that no fixture file can go stale. Shared by
// every test of the release scripts (plan 2026-11-04-lorren-distribution-step1-release, the ledger's pre-flight ruling).
import { createHash, generateKeyPairSync, sign } from 'node:crypto'

const raw = (publicKey) => publicKey.export({ format: 'der', type: 'spki' }).subarray(-32)

/** A fresh Ed25519 pair with an 8-byte key id. */
export function makeKey(keyId = Buffer.from('0102030405060708', 'hex')) {
  const { publicKey, privateKey } = generateKeyPairSync('ed25519')
  return { publicKey, privateKey, keyId }
}

/** The key's minisign public-key text. */
export function publicKeyText(key) {
  const body = Buffer.concat([Buffer.from('Ed', 'latin1'), key.keyId, raw(key.publicKey)])
  return `untrusted comment: minisign public key: ${key.keyId.toString('hex').toUpperCase()}\n${body.toString('base64')}\n`
}

/** A minisign signature text of `data` by `key`: prehashed ("ED", as Tauri's signer writes) unless told otherwise. */
export function signatureText(data, key, { prehashed = true, trusted = 'timestamp:1\tfile:x' } = {}) {
  const message = prehashed ? createHash('blake2b512').update(data).digest() : data
  const signature = sign(null, message, key.privateKey)
  const head = Buffer.concat([Buffer.from(prehashed ? 'ED' : 'Ed', 'latin1'), key.keyId, signature])
  const global = sign(null, Buffer.concat([signature, Buffer.from(trusted, 'utf8')]), key.privateKey)
  return `untrusted comment: signature from a test key\n${head.toString('base64')}\ntrusted comment: ${trusted}\n${global.toString('base64')}\n`
}

/** A key and both texts for `data`, as the plan names it. */
export function makeMinisign(data, { key = makeKey(), ...options } = {}) {
  return { key, publicKeyText: publicKeyText(key), signatureText: signatureText(data, key, options) }
}

/** Tauri's form of either text (what it writes to `.sig` and `.pub`, and puts in tauri.conf.json): base64 of the text. */
export const tauriForm = (text) => Buffer.from(text, 'utf8').toString('base64')
