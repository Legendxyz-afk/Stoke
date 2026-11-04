/* ============================================================================
   RELEASE -- npm run release -- <version> --notes <file> --key <file> [--channel beta] [--config <file>] [--master <dir>]

   The only way anything reaches Lorren's update feed (spec 2026-11-04 "The release script"; plan 2026-11-04 step 1, Task 4). In
   order, each refusal changing nothing: the config (no placeholder feed name), a clean tree, the version (a version, newer than the
   channel's), the notes, --deploy (it arrives with the Cloudflare project, step 4), the key (it must open); then the build with the
   release overlay (the config's public key and endpoint compiled in), the built installer's signature checked against the public
   key, the layout (feed.mjs), the whole-feed check and the deploy-drop check (check.mjs; on any problem the site is put back as it
   was), the deploy command this release needs (printed: deploying is step 4's), the local live check (the site served on
   127.0.0.1 and fetched as the updater and the website fetch it), and the tag lorren-v<version> (local; nothing is pushed).

   The master folder (default below) holds `site/` (what is deployed, whole) and `state/` (the last deploy's manifest). Back it up
   after every release: it holds installers that no later build reproduces byte for byte.
   ========================================================================== */
import { spawnSync } from 'node:child_process'
import { createHash } from 'node:crypto'
import { cpSync, existsSync, mkdtempSync, readFileSync, rmSync, statSync, writeFileSync } from 'node:fs'
import { createServer } from 'node:http'
import { tmpdir } from 'node:os'
import { basename, dirname, join, resolve, sep } from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'
import { checkFeed, dropped, manifest, readDeployed } from './check.mjs'
import { CHANNELS, INSTALLER, STABLE_COPY, channelState, layoutRelease, parseNotes } from './feed.mjs'
import { verify } from './minisign.mjs'
import { compareVersions, isVersion } from './semver.mjs'

const DESKTOP = resolve(dirname(fileURLToPath(import.meta.url)), '..', '..')
const TAURI = join(DESKTOP, 'node_modules', '@tauri-apps', 'cli', 'tauri.js')
export const DEFAULT_MASTER = 'M:\\New Drive\\Desktop\\Programming\\lorren-feed'
export const USAGE =
  'usage: npm run release -- <version> --notes <file> --key <file> [--channel beta] [--config <file>] [--master <dir>]'

const sha256 = (bytes) => createHash('sha256').update(bytes).digest('hex')
const withoutSlash = (url) => url.replace(/\/+$/, '')

/** The command line, as options; a malformed one throws USAGE. */
export function parseArgs(argv) {
  const out = { version: null, channel: 'stable', notes: null, key: null, config: 'release/release.config.json', master: DEFAULT_MASTER, deploy: false }
  const takes = { '--channel': 'channel', '--notes': 'notes', '--key': 'key', '--config': 'config', '--master': 'master' }
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i]
    if (arg === '--deploy') out.deploy = true
    else if (takes[arg]) {
      if (i + 1 >= argv.length) throw new Error(USAGE)
      out[takes[arg]] = argv[++i]
    } else if (arg.startsWith('--') || out.version !== null) throw new Error(USAGE)
    else out.version = arg
  }
  if (out.version === null) throw new Error(USAGE)
  return out
}

// The release config (paths in it are relative to desktop/): its feed, its public key's text, its Pages project.
function readConfig(path) {
  const file = resolve(DESKTOP, path)
  const config = JSON.parse(readFileSync(file, 'utf8'))
  if (JSON.stringify(config).includes('<feed>')) {
    throw new Error(`${basename(file)} still has the placeholder feed name: name the feed first (step 4)`)
  }
  for (const field of ['feedUrl', 'appPublicKey', 'pagesProject']) {
    if (typeof config[field] !== 'string' || config[field] === '') throw new Error(`${basename(file)} has no ${field}`)
  }
  return {
    feedUrl: config.feedUrl,
    appPublicKey: readFileSync(resolve(DESKTOP, config.appPublicKey), 'utf8').trim(),
    pagesProject: config.pagesProject,
    insecureHttp: config.insecureHttp === true,
  }
}

// The site served on a free 127.0.0.1 port, files only, and fetched as the updater (latest.json) and the website (the stable
// copy) fetch it. Returns the paths it fetched.
export async function liveCheck({ site, channel, version, installer }) {
  const root = resolve(site)
  const server = createServer((req, res) => {
    const parts = decodeURIComponent(new URL(req.url, 'http://local').pathname).split('/').filter(Boolean)
    const path = resolve(root, ...parts)
    if ((path !== root && !path.startsWith(root + sep)) || !existsSync(path) || !statSync(path).isFile()) {
      res.writeHead(404).end()
      return
    }
    res.writeHead(200).end(readFileSync(path))
  })
  await new Promise((ready) => server.listen(0, '127.0.0.1', ready))
  const base = `http://127.0.0.1:${server.address().port}`
  try {
    const fetched = []
    const manifestPath = `app/${channel}/latest.json`
    const answer = await fetch(`${base}/${manifestPath}`)
    const served = answer.ok ? await answer.json() : null
    if (served?.version !== version) throw new Error(`the served feed does not have ${version} in ${manifestPath}`)
    fetched.push(manifestPath)
    if (channel === 'stable') {
      const copyPath = `app/stable/${STABLE_COPY}`
      const copy = await fetch(`${base}/${copyPath}`)
      const bytes = copy.ok ? Buffer.from(await copy.arrayBuffer()) : Buffer.alloc(0)
      if (sha256(bytes) !== sha256(readFileSync(installer))) throw new Error(`the served feed does not have the built installer at ${copyPath}`)
      fetched.push(copyPath)
    }
    return fetched
  } finally {
    server.close()
  }
}

/**
 * One release. `deps`: git(args) -> stdout, openKey(keyPath, password) (throws if the key does not open), build({ version,
 * overlay, keyPath, password }) -> { installer, signature }, now() -> Date, password, and optionally log(line).
 * Returns { written, deployCommand, fetched }.
 */
export async function release(options, deps) {
  const { version, channel } = options
  const log = deps.log ?? (() => {})
  const config = readConfig(options.config)
  if (!CHANNELS.includes(channel)) throw new Error(`${channel} is not a channel`)
  if (deps.git(['status', '--porcelain']).trim() !== '') throw new Error('the tree is not clean')

  const site = join(options.master, 'site')
  const state = join(options.master, 'state')
  if (!isVersion(version)) throw new Error(`${version} is not a version`)
  const { latest, releases } = channelState(site, channel)
  if (latest?.version === version || releases.some((r) => r.version === version)) {
    throw new Error(`${version} is already in the ${channel} feed`)
  }
  if (latest && compareVersions(version, latest.version) <= 0) {
    throw new Error(`${version} is not newer than ${latest.version}, the ${channel} feed's newest`)
  }
  const notes = options.notes && existsSync(options.notes) ? parseNotes(readFileSync(options.notes, 'utf8')) : []
  if (notes.length === 0) throw new Error('no release notes')
  if (options.deploy) throw new Error('deploying arrives with the Cloudflare project (step 4)')
  if (!options.key) throw new Error('no --key: a release is signed with the private key, read from its file')
  // Absolute, for the build too: the Tauri CLI reads TAURI_SIGNING_PRIVATE_KEY from another folder, and a path it cannot find
  // there it takes for the key's own text ("Invalid symbol 46, offset 0": the "." of .keys; the test release of 2026-11-04).
  const keyPath = resolve(options.key)
  try {
    deps.openKey(keyPath, deps.password)
  } catch (e) {
    throw new Error(`the key does not open: ${e.message}`)
  }

  const updater = { pubkey: config.appPublicKey, endpoints: [`${withoutSlash(config.feedUrl)}/app/${channel}/latest.json`] }
  if (config.insecureHttp) updater.dangerousInsecureTransportProtocol = true
  const built = deps.build({ version, overlay: { version, plugins: { updater } }, keyPath, password: deps.password })
  const checked = verify(readFileSync(built.installer), built.signature, config.appPublicKey)
  if (!checked.ok) throw new Error(`the build's signature does not verify (${checked.reason}): nothing was laid out`)

  const scratch = mkdtempSync(join(tmpdir(), 'lorren-site-before-'))
  const before = join(scratch, 'site') // a path that does not exist yet: the copy is the site, not merged into a folder
  const hadSite = existsSync(site)
  if (hadSite) cpSync(site, before, { recursive: true })
  let laid
  try {
    laid = layoutRelease({ site, channel, version, notes, installer: built.installer, signature: built.signature, feedUrl: config.feedUrl, now: deps.now() })
    const problems = checkFeed({ site, feedUrl: config.feedUrl, appPublicKey: config.appPublicKey })
    if (problems.length > 0) throw new Error(`the feed does not check, so nothing was changed:\n  ${problems.join('\n  ')}`)
    const gone = dropped(readDeployed(state), manifest(site), laid.pruned)
    if (gone.length > 0) throw new Error(`this deploy would drop ${gone.join(', ')}, so nothing was changed`)
  } catch (e) {
    rmSync(site, { recursive: true, force: true })
    if (hadSite) cpSync(before, site, { recursive: true })
    throw e
  } finally {
    rmSync(scratch, { recursive: true, force: true })
  }

  const deployCommand = ['npx', 'wrangler', 'pages', 'deploy', site, '--project-name', config.pagesProject, '--branch', 'main']
  log(`the deploy this release needs (step 4 runs it): ${deployCommand.join(' ')}`)
  const fetched = await liveCheck({ site, channel, version, installer: built.installer })
  deps.git(['tag', `lorren-v${version}`])
  return { written: laid.written, deployCommand, fetched }
}

/** The real key check: signs a throwaway file with the key, as the build will. */
export function tauriOpenKey(keyPath, password) {
  const dir = mkdtempSync(join(tmpdir(), 'lorren-key-'))
  const probe = join(dir, 'probe.txt')
  writeFileSync(probe, 'does the key open?')
  try {
    // The password through the environment, which `tauri signer sign` reads as TAURI_SIGNING_PRIVATE_KEY_PASSWORD: a -p argument would
    // sit in the process list, readable by any process of the user, for as long as the signer runs.
    const env = { ...process.env, TAURI_SIGNING_PRIVATE_KEY_PASSWORD: password }
    const run = spawnSync(process.execPath, [TAURI, 'signer', 'sign', '-f', keyPath, probe], { encoding: 'utf8', env })
    if (run.status !== 0) throw new Error((run.stderr || run.stdout || `exit ${run.status}`).trim().split('\n').pop())
  } finally {
    rmSync(dir, { recursive: true, force: true })
  }
}

/** The real build: the per-user NSIS installer, signed by `tauri build` (it reads only these two variables: the M1 kit's finding). */
export function tauriBuild({ version, overlay, keyPath, password }) {
  const env = { ...process.env, TAURI_SIGNING_PRIVATE_KEY: keyPath, TAURI_SIGNING_PRIVATE_KEY_PASSWORD: password }
  const run = spawnSync(process.execPath, [TAURI, 'build', '--config', JSON.stringify(overlay)], { cwd: DESKTOP, env, stdio: 'inherit' })
  if (run.status !== 0) throw new Error(`tauri build failed (exit ${run.status})`)
  const installer = join(DESKTOP, 'src-tauri', 'target', 'release', 'bundle', 'nsis', INSTALLER(version))
  if (!existsSync(installer) || !existsSync(`${installer}.sig`)) throw new Error(`the build made no ${INSTALLER(version)} and its .sig`)
  return { installer, signature: readFileSync(`${installer}.sig`, 'utf8').trim() }
}

function git(args) {
  const run = spawnSync('git', args, { cwd: DESKTOP, encoding: 'utf8' })
  if (run.status !== 0) throw new Error(`git ${args.join(' ')}: ${(run.stderr || '').trim()}`)
  return run.stdout
}

async function main() {
  const options = parseArgs(process.argv.slice(2))
  const deps = { git, openKey: tauriOpenKey, build: tauriBuild, now: () => new Date(), password: process.env.LORREN_KEY_PASSWORD ?? '', log: console.log }
  const result = await release(options, deps)
  console.log(`released ${options.version} to the ${options.channel} feed in ${options.master} (${result.written.length} files written)`)
  console.log(`served and fetched: ${result.fetched.join(', ')}`)
}

if (process.argv[1] && pathToFileURL(resolve(process.argv[1])).href === import.meta.url) {
  main().catch((e) => {
    console.error(`release: ${e.message}`)
    process.exit(1)
  })
}
