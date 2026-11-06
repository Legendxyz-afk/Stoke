// Task 2 of plan 2026-11-04-lorren-distribution-step2-discover: the catalog script (build, validate, sign, write), with Hugging Face,
// the signer and the clock faked.
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { existsSync, mkdtempSync, readFileSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { buildCatalog, catalog, catalogText, parseCatalogArgs, validateEntry, validateModels } from '../catalog.mjs'
import { DEFAULT_MASTER } from '../release.mjs'
import { makeKey, publicKeyText, signatureText, tauriForm } from './keys.mjs'

const SOURCES = JSON.parse(readFileSync(resolve(dirname(fileURLToPath(import.meta.url)), '..', '..', '..', 'release', 'catalog.sources.json'), 'utf8'))
const QWEN = { size: 2497280736, sha256: '2fde00ce69dd4899c70d020845e2638353015bba0fdf161b3eb965f2bca4464e' }
const LLAMA = { size: 2019377696, sha256: '6c1a2b41161032677be168d354123594c0e6e67d2b9227c84f296ad037c728ff' }
const TREE = (repo, commit) => `https://huggingface.co/api/models/${repo}/tree/${commit}`
const QWEN_TREE = TREE('bartowski/Qwen_Qwen3-4B-Instruct-2507-GGUF', 'ae44f08e1392f39c0e474af10c3ff8355c8b6688')
const LLAMA_TREE = TREE('bartowski/Llama-3.2-3B-Instruct-GGUF', '5ab33fa94d1d04e903623ae72c95d1696f09f9e8')
const listing = (path, { size, sha256 }) => [
  { type: 'file', path: 'README.md', size: 100 },
  { type: 'file', path, size, lfs: { oid: sha256, size, pointerSize: 134 } },
]
const trees = () => ({
  [QWEN_TREE]: listing('Qwen_Qwen3-4B-Instruct-2507-Q4_K_M.gguf', QWEN),
  [LLAMA_TREE]: listing('Llama-3.2-3B-Instruct-Q4_K_M.gguf', LLAMA),
})
function fakeFetch(map = trees()) {
  const urls = []
  const fetchJson = async (url) => {
    urls.push(url)
    if (!(url in map)) throw new Error(`no such address: ${url}`)
    return map[url]
  }
  return { fetchJson, urls }
}
const NOW = new Date(Date.UTC(2026, 10, 6, 9, 30, 0))

test('the two sources build the catalog the table pins', async () => {
  const f = fakeFetch()
  const built = await buildCatalog(SOURCES, { fetchJson: f.fetchJson, now: () => NOW })
  assert.equal(built.format, 1)
  assert.equal(built.generated, '2026-11-06T09:30:00.000Z')
  assert.deepEqual(f.urls, [QWEN_TREE, LLAMA_TREE])
  assert.deepEqual(built.models, [
    {
      id: 'qwen3-4b-instruct-2507-q4_k_m', name: 'Qwen3-4B-Instruct-2507', version: 1,
      description: "Qwen3 4B Instruct, July 2025: Lorren's default, quick and capable on the processor.",
      architecture: 'qwen3', quantization: 'Q4_K_M', file: 'Qwen3-4B-Instruct-2507-Q4_K_M.gguf', ...QWEN,
      url: 'https://huggingface.co/bartowski/Qwen_Qwen3-4B-Instruct-2507-GGUF/resolve/ae44f08e1392f39c0e474af10c3ff8355c8b6688/Qwen_Qwen3-4B-Instruct-2507-Q4_K_M.gguf',
      licence: { name: 'Apache-2.0', url: 'https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/blob/main/LICENSE' },
      source: { repo: 'bartowski/Qwen_Qwen3-4B-Instruct-2507-GGUF', commit: 'ae44f08e1392f39c0e474af10c3ff8355c8b6688', path: 'Qwen_Qwen3-4B-Instruct-2507-Q4_K_M.gguf' },
      system_prompt: null, min_app_version: '0.1.0',
    },
    {
      id: 'llama-3.2-3b-instruct-q4_k_m', name: 'Llama-3.2-3B-Instruct', version: 1,
      description: "Meta's Llama 3.2 3B Instruct: smaller and quicker, for everyday chats.",
      architecture: 'llama', quantization: 'Q4_K_M', file: 'Llama-3.2-3B-Instruct-Q4_K_M.gguf', ...LLAMA,
      url: 'https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/5ab33fa94d1d04e903623ae72c95d1696f09f9e8/Llama-3.2-3B-Instruct-Q4_K_M.gguf',
      licence: { name: 'Llama 3.2 Community License', url: 'https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct/blob/main/LICENSE.txt' },
      source: { repo: 'bartowski/Llama-3.2-3B-Instruct-GGUF', commit: '5ab33fa94d1d04e903623ae72c95d1696f09f9e8', path: 'Llama-3.2-3B-Instruct-Q4_K_M.gguf' },
      system_prompt: null, min_app_version: '0.1.0',
    },
  ])
  assert.equal(catalogText(built), `${JSON.stringify(built, null, 2)}\n`)
})

test('a file that is not in the tree, or not an LFS file, or whose LFS size disagrees, is refused with its source named', async () => {
  const cases = [
    ['not in the tree', (m) => { m[QWEN_TREE] = [{ type: 'file', path: 'other.gguf', size: 1 }] }, /Qwen_Qwen3-4B-Instruct-2507-GGUF@ae44f08e.*Qwen_Qwen3-4B-Instruct-2507-Q4_K_M\.gguf is not in the repo at that commit/],
    ['not an LFS file', (m) => { delete m[LLAMA_TREE][1].lfs }, /Llama-3\.2-3B-Instruct-GGUF@5ab33fa9.*Llama-3\.2-3B-Instruct-Q4_K_M\.gguf is not an LFS file/],
    ['LFS size disagrees', (m) => { m[LLAMA_TREE][1].lfs.size += 1 }, /Llama-3\.2-3B-Instruct-GGUF@5ab33fa9.*size disagrees/],
  ]
  for (const [name, edit, pattern] of cases) {
    const map = trees()
    edit(map)
    await assert.rejects(buildCatalog(SOURCES, { fetchJson: fakeFetch(map).fetchJson, now: () => NOW }), pattern, name)
  }
})

test('one Error lists every problem, and a fetch failure names its source', async () => {
  const map = trees()
  map[QWEN_TREE] = []
  delete map[LLAMA_TREE]
  await assert.rejects(buildCatalog(SOURCES, { fetchJson: fakeFetch(map).fetchJson, now: () => NOW }), (e) => {
    assert.match(e.message, /is not in the repo/)
    assert.match(e.message, /Llama-3\.2-3B-Instruct-GGUF@5ab33fa9.*no such address/)
    return true
  })
})

const good = () => ({
  id: 'qwen3-4b-instruct-2507-q4_k_m', name: 'Qwen3-4B-Instruct-2507', version: 1, description: 'A model.', architecture: 'qwen3',
  quantization: 'Q4_K_M', file: 'Qwen3-4B-Instruct-2507-Q4_K_M.gguf', size: 2497280736, sha256: 'a'.repeat(64),
  url: 'https://huggingface.co/x/y/resolve/c/f.gguf', licence: { name: 'Apache-2.0', url: 'https://example.com/LICENSE' },
  source: { repo: 'x/y', commit: 'c', path: 'f.gguf' }, system_prompt: null, min_app_version: '0.1.0',
})

test('validateEntry: each rule', () => {
  assert.deepEqual(validateEntry(good()), [])
  const bad = [
    ['id', { id: 'Qwen' }], ['id', { id: '' }], ['id', { id: 'a'.repeat(65) }],
    ['name', { name: '' }], ['name', { name: 'n'.repeat(81) }],
    ['description', { description: 'd'.repeat(401) }],
    ['version', { version: 0 }], ['version', { version: 1.5 }],
    ['architecture', { architecture: 'qwen2vl' }],
    ['quantization', { quantization: '' }], ['quantization', { quantization: 'Q'.repeat(17) }],
    ['file', { file: '../x.gguf' }], ['file', { file: 'a..b.gguf' }], ['file', { file: 'dir/x.gguf' }], ['file', { file: 'x.bin' }],
    ['size', { size: 0 }], ['size', { size: 68719476737 }], ['size', { size: 1.5 }],
    ['sha256', { sha256: 'a'.repeat(63) }], ['sha256', { sha256: 'A'.repeat(64) }],
    ['url', { url: 'http://huggingface.co/x/y/resolve/c/f.gguf' }],
    ['licence.url', { licence: { name: 'L', url: 'http://example.com/LICENSE' } }],
    ['licence.name', { licence: { name: '', url: 'https://example.com/LICENSE' } }],
    ['min_app_version', { min_app_version: '1.0' }],
    ['system_prompt', { system_prompt: 's'.repeat(4001) }],
  ]
  for (const [field, change] of bad) {
    const problems = validateEntry({ ...good(), ...change })
    assert.equal(problems.length, 1, `${field} ${JSON.stringify(change)}: ${problems}`)
    assert.ok(problems[0].startsWith(`${field} `), problems[0])
  }
  // the edges that are allowed
  assert.deepEqual(validateEntry({ ...good(), size: 1, system_prompt: 's'.repeat(4000), file: 'A.GGUF', url: 'https://a.example/f' }), [])
  assert.deepEqual(validateEntry({ ...good(), size: 68719476736 }), [])
  assert.deepEqual(validateEntry({ ...good(), url: 'http://127.0.0.1:8766/f.gguf' }, { insecure: true }), [])
  assert.equal(validateEntry('nope').length, 1)
})

test('validateModels: duplicate ids and duplicate files, each named once', () => {
  const a = good()
  const b = { ...good(), file: 'Other.gguf' }
  const c = { ...good(), id: 'other', file: 'Other.gguf' }
  assert.deepEqual(validateModels([a, { ...good(), id: 'b', file: 'B.gguf' }]), [])
  assert.deepEqual(validateModels([a, b]), ['id "qwen3-4b-instruct-2507-q4_k_m" appears twice'])
  assert.deepEqual(validateModels([a, c, { ...c, id: 'third' }]), ['file "Other.gguf" appears twice'])
  assert.deepEqual(validateModels([a, { ...good(), id: 'b', size: 0, file: 'B.gguf' }]), ['models[1] (b): size is not a whole number from 1 to 68719476736'])
  assert.deepEqual(validateModels('x'), ['models is not a list'])
})

// ---- catalog(): the write ----
const catalogKey = makeKey(Buffer.from('c1c2c3c4c5c6c7c8', 'hex'))
const appKey = makeKey(Buffer.from('a1a2a3a4a5a6a7a8', 'hex'))

function setup() {
  const root = mkdtempSync(join(tmpdir(), 'lorren catalog '))
  writeFileSync(join(root, 'app.pub'), tauriForm(publicKeyText(appKey)))
  writeFileSync(join(root, 'catalog.pub'), tauriForm(publicKeyText(catalogKey)))
  const config = join(root, 'release.config.json')
  writeFileSync(config, JSON.stringify({
    feedUrl: 'http://127.0.0.1:8765', appPublicKey: join(root, 'app.pub'), catalogPublicKey: join(root, 'catalog.pub'),
    pagesProject: 'lorren-test', insecureHttp: true,
  }))
  const calls = { sign: [], signWith: catalogKey }
  const deps = {
    fetchJson: fakeFetch().fetchJson,
    now: () => NOW,
    password: 'test',
    sign: async ({ path, keyPath, password }) => {
      calls.sign.push({ path, keyPath, password })
      return signatureText(readFileSync(path), calls.signWith)
    },
  }
  const options = { key: join(root, 'catalog.key'), config, master: join(root, 'master'), deploy: false }
  return { root, deps, options, calls, dir: join(root, 'master', 'site', 'catalog') }
}

test("the catalog is signed, verified with the catalog key and written; the app key's signature is refused", async () => {
  const s = setup()
  const result = await catalog(s.options, s.deps)
  assert.ok(result.written.includes('catalog/catalog.json') && result.written.includes('catalog/catalog.json.minisig'))
  const text = readFileSync(join(s.dir, 'catalog.json'), 'utf8')
  assert.equal(text, catalogText(await buildCatalog(SOURCES, { fetchJson: fakeFetch().fetchJson, now: () => NOW })))
  assert.match(readFileSync(join(s.dir, 'catalog.json.minisig'), 'utf8'), /^untrusted comment:/)
  assert.equal(s.calls.sign.length, 1)
  assert.ok(s.calls.sign[0].keyPath.endsWith('catalog.key'))
  assert.equal(s.calls.sign[0].password, 'test')

  // a second run, signed with the app key: refused, and the first catalog stays
  const before = [readFileSync(join(s.dir, 'catalog.json'), 'utf8'), readFileSync(join(s.dir, 'catalog.json.minisig'), 'utf8')]
  s.calls.signWith = appKey
  s.deps.now = () => new Date(NOW.getTime() + 60000)
  await assert.rejects(catalog(s.options, s.deps), { message: "the catalog's signature does not verify with the catalog key (key id)" })
  assert.deepEqual([readFileSync(join(s.dir, 'catalog.json'), 'utf8'), readFileSync(join(s.dir, 'catalog.json.minisig'), 'utf8')], before)
})

test('an app-key catalog on a fresh master writes nothing', async () => {
  const s = setup()
  s.calls.signWith = appKey
  await assert.rejects(catalog(s.options, s.deps), /does not verify with the catalog key/)
  assert.equal(existsSync(s.dir), false)
})

test('--deploy is refused; a missing --key is refused; a config without a catalog key is refused', async () => {
  const s = setup()
  await assert.rejects(catalog({ ...s.options, deploy: true }, s.deps), { message: 'deploying arrives with the Cloudflare project (step 4)' })
  await assert.rejects(catalog({ ...s.options, key: null }, s.deps), { message: 'no --key: the catalog is signed with the catalog key, read from its file' })
  writeFileSync(s.options.config, JSON.stringify({ feedUrl: 'http://127.0.0.1:8765', appPublicKey: join(s.root, 'app.pub'), pagesProject: 'p' }))
  await assert.rejects(catalog(s.options, s.deps), /has no catalogPublicKey/)
  assert.equal(s.calls.sign.length, 0)
})

test('a catalog with a problem is refused before it is signed', async () => {
  const s = setup()
  const map = trees()
  map[QWEN_TREE] = []
  s.deps.fetchJson = fakeFetch(map).fetchJson
  await assert.rejects(catalog(s.options, s.deps), /is not in the repo/)
  assert.equal(s.calls.sign.length, 0)
})

test('parseCatalogArgs: the options, their defaults, and a usage error', () => {
  assert.deepEqual(parseCatalogArgs([]), { key: null, config: 'release/release.config.json', master: DEFAULT_MASTER, deploy: false })
  assert.deepEqual(parseCatalogArgs(['--key', 'k', '--config', 'c', '--master', 'm', '--deploy']), { key: 'k', config: 'c', master: 'm', deploy: true })
  assert.throws(() => parseCatalogArgs(['--key']), /usage/)
  assert.throws(() => parseCatalogArgs(['stray']), /usage/)
})
