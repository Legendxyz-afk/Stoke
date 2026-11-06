# Releasing Lorren

`npm run release` is the only way anything reaches Lorren's update feed (spec
`docs/superpowers/specs/2026-11-04-lorren-distribution-first-design.md`). It builds the signed installer, lays it out in the feed's
master folder, checks the whole feed, shows the deploy it needs, serves the feed on this PC and fetches it the way the app and the
website will, and tags the commit `lorren-v<version>` (locally; pushing is a separate decision).

Until step 4 of the spec (the Cloudflare project and the real keys), `release.config.json` holds placeholders and the command
refuses to use it: only the test configuration below works.

## The keys (once, by ghost)

Two key pairs, one for the app and one for the model catalog. Make each on this PC, with the USB stick that will keep it plugged in:

```
npm run tauri signer generate -- -w E:\lorren-app.key
npm run tauri signer generate -- -w E:\lorren-catalog.key
```

Each asks for a password: keep it with the stick, not on the PC. The `.key` files never come into this repository or the feed's
folder. Copy `E:\lorren-app.key.pub` to `desktop/release/keys/app.pub` (a public key is public by nature: it is committed) and the
catalog's to `desktop/release/keys/catalog.pub`.

## The master folder

`M:\New Drive\Desktop\Programming\lorren-feed` (another with `--master`):

- `site/` is what is deployed, whole, every time (a Pages deploy replaces the site, so nothing else may ever be deployed there);
- `state/` is not deployed: it remembers what the last deploy held, so that no deploy can drop a file it did not mean to.

Back the folder up after every release. It holds the installers that shipped, and no later build reproduces them byte for byte.

## A release

```
set LORREN_KEY_PASSWORD=<the key's password>
npm run release -- 0.1.0 --notes notes.txt --key E:\lorren-app.key
npm run release -- 0.1.1-beta.1 --channel beta --notes notes.txt --key E:\lorren-app.key
```

`notes.txt` has one note per line (`- ` or `* ` bullets are fine); they become the Updates page's list and the app's update notes.

## The model catalog

The catalog is the list of models Lorren's Discover screen offers: `release/catalog.sources.json` names each by a Hugging Face repo,
a commit and a path. It is signed with the CATALOG key, which is not the app's key: a leaked catalog key must not be able to ship
code. Make it once, with its own file and password, kept offline like the app key (the same command, a different file):

```
npm run tauri signer generate -- -w E:\lorren-catalog.key
```

Copy `E:\lorren-catalog.key.pub` to `desktop/release/keys/catalog.pub` (that is where `release.config.json` looks for it; until
the file is there, `npm run release` and `npm run catalog` refuse the config). Then:

```
set LORREN_KEY_PASSWORD=<the catalog key's password>
npm run catalog -- --key E:\lorren-catalog.key
```

It reads Hugging Face (reads only) for each source's commit, takes the size and sha256 of each file from the listing, signs the
catalog, checks the signature with the catalog's public key and writes `catalog/catalog.json` and `catalog/catalog.json.minisig`
into the master folder's `site/`. It refuses, writing nothing, when a file is not in the repo at that commit, is not an LFS file,
has sizes that disagree, breaks an entry rule (ids, file names, sizes, sha256, https addresses, licence, versions, duplicates), or
the signature does not verify with the catalog key (a catalog signed with the app key, for one).

Run `npm run catalog` before the first release: with `catalogPublicKey` in the config, the feed check wants a signed, valid catalog
in the feed (`catalog: catalog/catalog.json is missing` otherwise), and the build compiles in its address and the public key. The
test config (`release.test.json`) uses `.keys/catalog.key` (made by `python scripts/update-feed.py keys`, password `lorren-test`).

## A test release (the M1 test key, a local feed, any time)

```
set LORREN_KEY_PASSWORD=lorren-test
npm run release -- 0.0.3 --channel beta --config release/release.test.json --notes notes.txt --key .keys/test.key --master %TEMP%\lorren-feed-test
```

It builds the app in release mode (minutes), so run it on a free PC. Afterwards delete its tag: `git tag -d lorren-v0.0.3`.

## When it refuses

Every refusal comes before anything changes, or puts the folder back as it was.

| it says | what to do |
|---|---|
| still has the placeholder feed name | step 4: name the feed and fill in `release.config.json` |
| the tree is not clean | commit or set aside what is changed: a release is built from a commit |
| is already in the ... feed / is not newer than | pick the next version |
| no release notes | write at least one line in the notes file |
| deploying arrives with the Cloudflare project | `--deploy` comes in step 4 |
| the key does not open | the key path or `LORREN_KEY_PASSWORD` is wrong |
| the build's signature does not verify | the key is not the one whose `.pub` the config names |
| catalog: ... is missing / fails its signature | run `npm run catalog`: the catalog is signed with the catalog key and checked with the catalog's public key |
| the feed does not check | the master folder was changed by hand: the list says what is wrong |
| this deploy would drop | a file the feed served is gone from the folder: restore it from the backup |
| the served feed does not have | the folder and the release disagree: run the tests (`npm run test:release`) and look |
