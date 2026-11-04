/* ============================================================================
   MINISIGN -- checks a minisign signature the way Lorren's updater does (tauri-plugin-updater, through the minisign-verify
   crate), with nothing but Node's crypto: Ed25519, and BLAKE2b-512 for the prehashed "ED" form that Tauri's signer writes.
   The release command checks every installer it builds with this before laying it out, and the whole-feed check checks every
   installer in the feed, so that nothing the updater would refuse is ever deployed (plan 2026-11-04 step 1, Task 1).

   A public key's text is "untrusted comment: ..." then base64("Ed" | key id (8 bytes) | key (32)). A signature's text is
   "untrusted comment: ...", base64(algorithm (2) | key id (8) | signature (64)), "trusted comment: <text>", base64(global
   signature (64)). "ED" signs BLAKE2b-512 of the data, "Ed" the data itself; the global signature signs signature | the trusted
   comment. Tauri keeps both texts base64-encoded (the .sig and .pub files, tauri.conf.json): either form is read.
   ========================================================================== */
import { createHash, createPublicKey, verify as ed25519 } from 'node:crypto'

const SPKI_ED25519 = Buffer.from('302a300506032b6570032100', 'hex') // the DER prefix that makes 32 raw bytes an Ed25519 public key

// The text's lines, from either form, with a BOM and Windows line ends let go (but a line's own content left exactly as it is: the
// trusted comment is signed byte for byte).
function lines(text) {
  let t = String(text ?? '').replace(/^﻿/, '').trim()
  if (!t.startsWith('untrusted comment:')) t = Buffer.from(t, 'base64').toString('utf8').replace(/^﻿/, '').trim()
  return t.split('\n').map((line) => line.replace(/\r$/, ''))
}

/** A minisign public key: its 8-byte id and its 32-byte Ed25519 key; null when the text is not one. */
export function parsePublicKey(text) {
  const [comment, body] = lines(text)
  if (!comment?.startsWith('untrusted comment:') || !body) return null
  const raw = Buffer.from(body, 'base64')
  if (raw.length !== 42 || raw.subarray(0, 2).toString('latin1') !== 'Ed') return null
  return { keyId: raw.subarray(2, 10), key: raw.subarray(10) }
}

/** A minisign signature: its algorithm, key id, signature, trusted comment and global signature; null when the text is not one. */
export function parseSignature(text) {
  const [comment, body, trusted, global] = lines(text)
  if (!comment?.startsWith('untrusted comment:') || !body || !trusted?.startsWith('trusted comment: ') || !global) return null
  const raw = Buffer.from(body, 'base64')
  const algorithm = raw.subarray(0, 2).toString('latin1')
  const globalSignature = Buffer.from(global, 'base64')
  if (raw.length !== 74 || (algorithm !== 'Ed' && algorithm !== 'ED') || globalSignature.length !== 64) return null
  return {
    algorithm,
    keyId: raw.subarray(2, 10),
    signature: raw.subarray(10),
    trustedComment: trusted.slice('trusted comment: '.length),
    globalSignature,
  }
}

/** Whether `signatureText` is a good signature of `data` by `publicKeyText`: { ok: true }, or { ok: false, reason } saying why. */
export function verify(data, signatureText, publicKeyText) {
  const pk = parsePublicKey(publicKeyText)
  const sig = parseSignature(signatureText)
  if (!pk || !sig) return { ok: false, reason: 'format' }
  if (!pk.keyId.equals(sig.keyId)) return { ok: false, reason: 'key id' }
  let key
  try {
    key = createPublicKey({ key: Buffer.concat([SPKI_ED25519, pk.key]), format: 'der', type: 'spki' })
  } catch {
    return { ok: false, reason: 'format' }
  }
  const message = sig.algorithm === 'ED' ? createHash('blake2b512').update(data).digest() : data
  if (!ed25519(null, message, key, sig.signature)) return { ok: false, reason: 'signature' }
  const signed = Buffer.concat([sig.signature, Buffer.from(sig.trustedComment, 'utf8')])
  if (!ed25519(null, signed, key, sig.globalSignature)) return { ok: false, reason: 'trusted comment' }
  return { ok: true }
}
