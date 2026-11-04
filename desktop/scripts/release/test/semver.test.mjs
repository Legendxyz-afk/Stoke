// Task 2 of plan 2026-11-04-lorren-distribution-step1-release: versions, ordered as the updater orders them (SemVer 2.0.0).
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { compareVersions, isVersion } from '../semver.mjs'

test('versions are ordered as SemVer orders them', () => {
  const below = [
    ['0.1.0', '0.1.1'],
    ['0.1.0-beta.1', '0.1.0'], // a pre-release comes before its release
    ['0.1.0-beta.2', '0.1.0-beta.10'], // numeric identifiers compare as numbers
    ['0.1.0-alpha', '0.1.0-beta'], // the others as text
    ['0.9.0', '0.10.0'],
    ['0.1.0-beta', '0.1.0-beta.1'], // more identifiers, all else equal, come later
    ['0.1.0-1', '0.1.0-alpha'], // a numeric identifier comes before a text one
  ]
  for (const [a, b] of below) {
    assert.equal(Math.sign(compareVersions(a, b)), -1, `${a} < ${b}`)
    assert.equal(Math.sign(compareVersions(b, a)), 1, `${b} > ${a}`)
  }
  assert.equal(compareVersions('1.0.0', '1.0.0'), 0)
  assert.equal(compareVersions('1.0.0-rc.1', '1.0.0-rc.1'), 0)
})

test('a version is x.y.z with an optional pre-release, and nothing else', () => {
  for (const v of ['0.1.0', '10.20.30', '0.1.0-beta.1', '1.0.0-rc.1', '0.0.3']) assert.equal(isVersion(v), true, v)
  for (const v of ['0.1', 'v0.1.0', '0.1.0+build', '01.0.0', '0.1.0-', '0.1.0-beta..1', '', '0.1.0-01', ' 0.1.0']) {
    assert.equal(isVersion(v), false, JSON.stringify(v))
  }
})
