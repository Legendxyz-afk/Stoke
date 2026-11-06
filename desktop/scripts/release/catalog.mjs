/* ============================================================================
   CATALOG -- npm run catalog -- --key <file> [--config <file>] [--master <dir>]

   Makes the model catalog the app's Discover screen reads (spec 2026-11-04 section 3.7; plan 2026-11-04 step 2, Task 2). The
   sources (release/catalog.sources.json) name each model by a Hugging Face repo, a commit and a path; this asks Hugging Face's
   tree API for that commit (reads only), takes each file's size and sha256 from the listing (an LFS file's oid IS its sha256),
   checks every entry against the entry rules (check.mjs, the same list as the app's), signs the catalog with the CATALOG key (a
   key of its own: the app's update key must not be able to sign a catalog), checks the signature with the catalog's public key
   and only then writes site/catalog/catalog.json and catalog.json.minisig into the master folder. Any refusal changes nothing.

   The password goes to the signer through LORREN_KEY_PASSWORD, never on a command line. --deploy arrives with step 4.
   ========================================================================== */
import { spawnSync } from 'node:child_process'
import { copyFileSync, existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'
import { checkCatalog, validateEntry, validateModels } from './check.mjs'
import { DEFAULT_MASTER, readConfig } from './release.mjs'
import { verify } from './minisign.mjs'

export { validateEntry, validateModels }

const DESKTOP = resolve(dirname(fileURLToPath(import.meta.url)), '..', '..')
const TAURI = join(DESKTOP, 'node_modules', '@tauri-apps', 'cli', 'tauri.js')
const SOURCES = join(DESKTOP, 'release', 'catalog.sources.json')
const HF = 'https://huggingface.co'
export const USAGE = 'usage: npm run catalog -- --key <file> [--config <file>] [--master <dir>]'

/** The command line, as options; a malformed one throws USAGE. */
export function parseCatalogArgs(argv) {
  const out = { key: null, config: 'release/release.config.json', master: DEFAULT_MASTER, deploy: false }
  const takes = { '--key': 'key', '--config': 'config', '--master': 'master' }
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i]
    if (arg === '--deploy') out.deploy = true
    else if (takes[arg]) {
      if (i + 1 >= argv.length) throw new Error(USAGE)
      out[takes[arg]] = argv[++i]
    } else throw new Error(USAGE)
  }
  return out
}

/** The catalog as the file holds it. */
export const catalogText = (catalog) => `${JSON.stringify(catalog, null, 2)}\n`

/**
 * The catalog the sources describe. `deps`: fetchJson(url) -> the parsed answer, now() -> Date. Throws ONE Error listing every
 * problem (a source Hugging Face cannot answer for, a file that is not there, and every entry rule broken).
 */
export async function buildCatalog(sources, { fetchJson, now }) {
  const problems = []
  const trees = new Map()
  const models = []
  for (const source of sources) {
    const { repo, commit, path } = source.source
    const where = `${repo}@${commit.slice(0, 8)}`
    const treeUrl = `${HF}/api/models/${repo}/tree/${commit}`
    if (!trees.has(treeUrl)) {
      try {
        trees.set(treeUrl, await fetchJson(treeUrl))
      } catch (e) {
        trees.set(treeUrl, e)
      }
    }
    const tree = trees.get(treeUrl)
    if (tree instanceof Error) {
      problems.push(`${where}: ${tree.message}`)
      continue
    }
    const file = Array.isArray(tree) ? tree.find((item) => item.path === path) : undefined
    if (!file) problems.push(`${where}: ${path} is not in the repo at that commit`)
    else if (!file.lfs || typeof file.lfs.oid !== 'string') problems.push(`${where}: ${path} is not an LFS file, so it has no sha256 to take`)
    else if (file.lfs.size !== file.size) problems.push(`${where}: ${path}'s size disagrees with its LFS size (${file.size} and ${file.lfs.size})`)
    else {
      models.push({
        id: source.id, name: source.name, version: source.version, description: source.description,
        architecture: source.architecture, quantization: source.quantization, file: source.file,
        size: file.size, sha256: file.lfs.oid,
        url: `${HF}/${repo}/resolve/${commit}/${path}`,
        licence: source.licence, source: source.source,
        system_prompt: source.system_prompt ?? null, min_app_version: source.min_app_version,
      })
    }
  }
  for (const model of models) {
    if (!model.url.startsWith(`${HF}/`)) problems.push(`${model.id}: url is not a ${HF}/ address`)
  }
  problems.push(...validateModels(models))
  if (problems.length > 0) throw new Error(`the catalog does not build:\n  ${problems.join('\n  ')}`)
  return { format: 1, generated: now().toISOString(), models }
}

/** The real signer: `tauri signer sign` with the catalog key; returns the minisign TEXT (the CLI writes it base64-encoded in <path>.sig, which is deleted). */
export function signCatalog({ path, keyPath, password }) {
  const env = { ...process.env, TAURI_SIGNING_PRIVATE_KEY_PASSWORD: password }
  const run = spawnSync(process.execPath, [TAURI, 'signer', 'sign', '-f', keyPath, path], { encoding: 'utf8', env })
  const sigPath = `${path}.sig`
  if (run.status !== 0 || !existsSync(sigPath)) {
    throw new Error(`the catalog key did not sign: ${(run.stderr || run.stdout || `exit ${run.status}`).trim().split('\n').pop()}`)
  }
  try {
    return Buffer.from(readFileSync(sigPath, 'utf8').trim(), 'base64').toString('utf8')
  } finally {
    rmSync(sigPath, { force: true })
  }
}

/**
 * One catalog run. `deps`: fetchJson(url), now(), sign({ path, keyPath, password }) -> minisign text, password, optionally log(line).
 * Returns { written: ['catalog/catalog.json', 'catalog/catalog.json.minisig'], models }.
 */
export async function catalog(options, deps) {
  const log = deps.log ?? (() => {})
  if (options.deploy) throw new Error('deploying arrives with the Cloudflare project (step 4)')
  if (!options.key) throw new Error('no --key: the catalog is signed with the catalog key, read from its file')
  const config = readConfig(options.config)
  if (!config.catalogPublicKey) throw new Error(`${options.config} has no catalogPublicKey: the catalog cannot be checked without it`)
  const keyPath = resolve(options.key)

  const built = await buildCatalog(JSON.parse(readFileSync(SOURCES, 'utf8')), deps)
  const bytes = Buffer.from(catalogText(built), 'utf8')

  const scratch = mkdtempSync(join(tmpdir(), 'lorren-catalog-'))
  try {
    const path = join(scratch, 'catalog.json')
    writeFileSync(path, bytes)
    const signature = await deps.sign({ path, keyPath, password: deps.password })
    const checked = verify(bytes, signature, config.catalogPublicKey)
    if (!checked.ok) throw new Error(`the catalog's signature does not verify with the catalog key (${checked.reason})`)
    const dir = join(options.master, 'site', 'catalog')
    mkdirSync(dir, { recursive: true })
    copyFileSync(path, join(dir, 'catalog.json'))
    writeFileSync(join(dir, 'catalog.json.minisig'), signature)
  } finally {
    rmSync(scratch, { recursive: true, force: true })
  }
  const problems = checkCatalog({ site: join(options.master, 'site'), catalogPublicKey: config.catalogPublicKey })
  if (problems.length > 0) throw new Error(`the catalog was written but does not check:\n  ${problems.join('\n  ')}`)
  log(`the catalog: ${built.models.length} models, signed and checked`)
  return { written: ['catalog/catalog.json', 'catalog/catalog.json.minisig'], models: built.models }
}

async function fetchJson(url) {
  const answer = await fetch(url, { headers: { 'User-Agent': 'lorren-catalog' } })
  if (!answer.ok) throw new Error(`${url} answered ${answer.status}`)
  return answer.json()
}

async function main() {
  const options = parseCatalogArgs(process.argv.slice(2))
  const deps = { fetchJson, now: () => new Date(), sign: signCatalog, password: process.env.LORREN_KEY_PASSWORD ?? '', log: console.log }
  const result = await catalog(options, deps)
  for (const m of result.models) console.log(`${m.id}: ${m.size} bytes, sha256 ${m.sha256}`)
  console.log(`wrote ${result.written.join(' and ')} in ${join(options.master, 'site')}`)
}

if (process.argv[1] && pathToFileURL(resolve(process.argv[1])).href === import.meta.url) {
  main().catch((e) => {
    console.error(`catalog: ${e.message}`)
    process.exit(1)
  })
}
