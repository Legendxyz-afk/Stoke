// Task 4 of plan 2026-11-04-lorren-distribution-step1-release: the release command, with git, the key, the build and the clock
// faked (the real build compiles the app: Task 5 runs it once).
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { existsSync, mkdirSync, mkdtempSync, readFileSync, readdirSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { DEFAULT_MASTER, USAGE, parseArgs, release } from '../release.mjs'
import { makeKey, publicKeyText, signatureText, tauriForm } from './keys.mjs'

const key = makeKey()
const otherKey = makeKey(Buffer.from('ffeeddccbbaa9988', 'hex'))
const PUB = tauriForm(publicKeyText(key))

// A master folder (its name has a space: Review Focus 2), a config, a notes file and the fakes; `calls` records what they were asked.
function setup({ feedUrl = 'http://127.0.0.1:8765', pagesProject = 'lorren-test' } = {}) {
  const root = mkdtempSync(join(tmpdir(), 'lorren feed '))
  writeFileSync(join(root, 'app.pub'), PUB)
  const config = join(root, 'release.config.json')
  writeFileSync(config, JSON.stringify({ feedUrl, appPublicKey: join(root, 'app.pub'), pagesProject, insecureHttp: true }))
  writeFileSync(join(root, 'notes.txt'), '- The first public version\n')
  const calls = { git: [], build: [], dirty: false, signWith: key }
  const deps = {
    git: (args) => {
      calls.git.push(args)
      return calls.dirty && args[0] === 'status' ? ' M src/x.ts\n' : ''
    },
    openKey: () => {},
    build: ({ version, overlay }) => {
      calls.build.push(overlay)
      const installer = join(root, `built-${version}.exe`)
      writeFileSync(installer, `MZ built ${version}`)
      return { installer, signature: tauriForm(signatureText(readFileSync(installer), calls.signWith)) }
    },
    now: () => new Date(2026, 10, 20, 12),
    password: 'test',
  }
  const options = {
    version: '0.1.0', channel: 'stable', notes: join(root, 'notes.txt'), key: join(root, 'app.key'), config,
    master: join(root, 'master'), deploy: false,
  }
  return { root, deps, options, calls, site: join(root, 'master', 'site') }
}

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

test('a stable release lays out, checks, plans the deploy, serves and tags', async () => {
  const s = setup()
  const result = await release(s.options, s.deps)
  for (const file of ['Lorren_0.1.0_x64-setup.exe', 'Lorren_0.1.0_x64-setup.exe.sig', 'latest.json', 'releases.json', 'Lorren-setup.exe']) {
    assert.ok(result.written.includes(`app/stable/${file}`), file)
  }
  assert.deepEqual(result.deployCommand, ['npx', 'wrangler', 'pages', 'deploy', s.site, '--project-name', 'lorren-test', '--branch', 'main'])
  assert.deepEqual(result.fetched, ['app/stable/latest.json', 'app/stable/Lorren-setup.exe'])
  assert.deepEqual(s.calls.git.at(-1), ['tag', 'lorren-v0.1.0'])
  assert.deepEqual(s.calls.build, [{
    version: '0.1.0',
    plugins: { updater: { pubkey: PUB, endpoints: ['http://127.0.0.1:8765/app/stable/latest.json'], dangerousInsecureTransportProtocol: true } },
  }])
})

test('an https feed gets no insecure-transport flag in the build', async () => {
  const s = setup()
  writeFileSync(s.options.config, JSON.stringify({ feedUrl: 'https://lorren-test.pages.dev', appPublicKey: join(s.root, 'app.pub'), pagesProject: 'lorren-test' }))
  // the live check serves locally and does not read the feed's address, so the release goes through to its end
  await release(s.options, s.deps)
  assert.deepEqual(s.calls.build[0].plugins.updater, { pubkey: PUB, endpoints: ['https://lorren-test.pages.dev/app/stable/latest.json'] })
})

test('a beta release serves and fetches its own manifest only', async () => {
  const s = setup()
  const result = await release({ ...s.options, version: '0.1.0-beta.1', channel: 'beta' }, s.deps)
  assert.deepEqual(result.fetched, ['app/beta/latest.json'])
  assert.equal(existsSync(join(s.site, 'app', 'stable')), false)
})

test('a dirty tree is refused before anything is built', async () => {
  const s = setup()
  s.calls.dirty = true
  await assert.rejects(release(s.options, s.deps), { message: 'the tree is not clean' })
  assert.equal(s.calls.build.length, 0)
  assert.equal(existsSync(s.site), false)
})

test('the placeholder feed name is refused', async () => {
  const s = setup({ feedUrl: 'https://<feed>.pages.dev', pagesProject: '<feed>' })
  await assert.rejects(release(s.options, s.deps), {
    message: 'release.config.json still has the placeholder feed name: name the feed first (step 4)',
  })
  assert.equal(s.calls.build.length, 0)
})

test('a key that does not open is refused before the build', async () => {
  const s = setup()
  s.deps.openKey = () => { throw new Error('wrong password') }
  await assert.rejects(release(s.options, s.deps), { message: 'the key does not open: wrong password' })
  assert.equal(s.calls.build.length, 0)
})

test('a build whose signature does not verify lays out nothing', async () => {
  const s = setup()
  s.calls.signWith = otherKey
  await assert.rejects(release(s.options, s.deps), { message: "the build's signature does not verify (key id): nothing was laid out" })
  assert.equal(existsSync(s.site), false)
})

test('a failed check puts the site back as it was', async () => {
  const s = setup()
  await release(s.options, s.deps)
  mkdirSync(join(s.site, 'app', 'beta'))
  writeFileSync(join(s.site, 'app', 'beta', 'latest.json'), '{}') // a channel that fails the check
  const before = snapshot(s.site)
  await assert.rejects(release({ ...s.options, version: '0.1.1' }, s.deps), /app\/beta: latest\.json does not parse/)
  assert.deepEqual(snapshot(s.site), before)
})

test('the same version twice is refused, and tagged once', async () => {
  const s = setup()
  await release(s.options, s.deps)
  await assert.rejects(release(s.options, s.deps), { message: '0.1.0 is already in the stable feed' })
  assert.equal(s.calls.git.filter((args) => args[0] === 'tag').length, 1)
  assert.equal(s.calls.build.length, 1)
})

test('notes with nothing in them are refused before the build', async () => {
  const s = setup()
  writeFileSync(s.options.notes, '\n- \n')
  await assert.rejects(release(s.options, s.deps), { message: 'no release notes' })
  assert.equal(s.calls.build.length, 0)
})

test('--deploy is refused in this step, before anything is built', async () => {
  const s = setup()
  await assert.rejects(release({ ...s.options, deploy: true }, s.deps), { message: 'deploying arrives with the Cloudflare project (step 4)' })
  assert.equal(s.calls.build.length, 0)
})

test('the arguments: their defaults, and a missing version', () => {
  assert.deepEqual(parseArgs(['0.1.0', '--notes', 'n.txt', '--key', 'k']), {
    version: '0.1.0', channel: 'stable', notes: 'n.txt', key: 'k', config: 'release/release.config.json', master: DEFAULT_MASTER,
    deploy: false,
  })
  assert.equal(parseArgs(['0.1.0-beta.1', '--channel', 'beta']).channel, 'beta')
  assert.equal(parseArgs(['0.1.0', '--deploy']).deploy, true)
  assert.throws(() => parseArgs(['--notes', 'n.txt']), { message: USAGE })
  assert.throws(() => parseArgs(['0.1.0', '--notes']), { message: USAGE })
  assert.throws(() => parseArgs(['0.1.0', '--what']), { message: USAGE })
})
