# Lorren: distribution first (design)

2026-11-04. Ghost: "i want the download to work properly and sooner", after agreeing with the Download button as designed (a fixed
address that always holds the newest installer, the version, size and date beside the button, download counts from Cloudflare's own
numbers rather than tracking on the site), then "start working". This changes the ORDER of the milestones in
`2026-09-25-lorren-desktop-design.md` §11 and adds the website's half of the download. Everything that spec says about updates, the
catalog, signing and hosting (§3.7) stands; where the two differ, this one is newer.

## Why

The download is the product's front door, and today it is "Coming soon": the feed, the release script, the real keys and Discover
are all in milestone 4, behind the complete chat (2) and the complete console (3). And a file on a server is not yet a download that
works: a stranger who installs today's build finds no models (the library looks in Kabir's own folders, Downloads and Ollama's store)
and has no way to get one inside the app.

## The decision

**Distribution moves up to come right after milestone 1.** The first public version, **v0.1.0, is milestone 1 plus distribution**:
the Control Room and the Chat as milestone 1 has them, Discover with downloads that feel like any download, and signed updates from
the real feed. The complete chat and the complete console follow as ordinary automatic updates (v0.2, v0.3), which is what the
updater exists for.

What v0.1 does not have is not claimed: every "today" badge on the website is checked against the v0.1 build before the button goes
live, and whatever only milestones 2 and 3 bring is marked "coming".

## How the download works

**The feed** (one Cloudflare Pages project of its own, `<feed>.pages.dev`, deployed by direct upload; §3.7):

| path | what |
|---|---|
| `/app/stable/Lorren-setup.exe` | a byte copy of the newest stable installer: **the website's fixed link**, never versioned |
| `/app/stable/Lorren_<version>_x64-setup.exe` and `.sig` | the versioned installer and its signature (the updater's) |
| `/app/stable/latest.json` | the updater's manifest (version, notes, date, the versioned URL and its signature) |
| `/app/stable/releases.json` | `[{ version, date, notes }]`, newest first: the website's Updates page (the contract `company/src/site/releases.js` already reads) |
| `/app/beta/...` | the same for the beta channel (the website never links to it) |
| `/catalog/catalog.json` and `.minisig` | the model catalog, signed with its own offline key |
| `/_headers` | `.exe` files download as attachments; `latest.json`, `releases.json`, `Lorren-setup.exe` and the catalog are revalidated on every request (a release is seen at once); versioned files are immutable |
| `/index.html` | one plain line saying what this address is, so the root is not a 404 |

The last two versions of each channel stay in the feed, so an update that started downloading just before a release still finishes.

**The release script** (`npm run release -- <version> [--channel beta]`, in `desktop/`) is the only way anything reaches the feed:
1. refuses to start unless the tree is clean, the version is newer than the feed's, the release notes exist and the private key
   opens (it is read from the USB, never copied into the repo);
2. builds the per-user NSIS installer with the release configuration (the real public key and the real endpoint compiled in);
3. signs it, then **verifies the signature with the public key** before going on;
4. lays out the channel in the feed's local master folder (the versioned files, `latest.json`, the stable copy, `releases.json`,
   pruning to two versions);
5. checks the WHOLE folder before deploying (a deploy replaces the site: both channels and the catalog must be complete and every
   manifest must point at a file that is there and whose signature verifies), then deploys it with `wrangler pages deploy`;
6. checks the live feed afterwards: `latest.json` names the new version, and the stable `Lorren-setup.exe` downloaded from the live
   address has the sha256 of the file it built;
7. tags the commit `lorren-v<version>` (local; pushing stays ghost's call).

**The catalog script** (`npm run catalog`) builds `catalog.json` from a short source list (Hugging Face files pinned to a commit),
takes each file's size and sha256 from Hugging Face's API (the LFS metadata: no model is downloaded to build the catalog), signs it
with the catalog key, and writes it into the same master folder; the release script deploys it with everything else.

**The website** (`company`, branch `lorren-site`):
- the button points at `https://<feed>.pages.dev/app/stable/Lorren-setup.exe` (`VITE_DOWNLOAD_URL`), the Updates page at the feed
  (`VITE_FEED_URL`; already built);
- beside the button: the version, the size and the date (`v0.1.0 · 6.2 MB · 20 November 2026`), read from the feed when the site is
  built, as the Updates page already is; a release that ships before the site is rebuilt still downloads the newest file (the link is
  fixed), only the line beside it is a release behind until the next build;
- after the click, a short note where the button was: the file is downloading, open it when it is done, and what the two warnings
  are (Chrome's "isn't commonly downloaded": keep it; Windows' "Windows protected your PC": More info, then Run anyway), because
  Lorren is new and not code-signed (my call: this is the moment the warnings matter, so that is when they are shown);
- on a phone, as now: Lorren is for Windows PCs, copy the link;
- the download count: Cloudflare's own numbers for the feed project. If Pages does not count requests per file on a `pages.dev`
  project, the builder adds the smallest thing that does (for example a `/download` route on the feed that redirects to the file and
  counts in Workers Analytics Engine), never a script on the site and never anything personal. To be settled by looking first.

## "Properly": what has to be true before v0.1.0 goes on the button

1. On a **fresh Windows user account**, a real browser download from the website's button (live feed): the installer runs with no
   admin prompt, Lorren opens, Discover lists the catalog, a model downloads (and resumes after the connection drops), and a chat
   streams.
2. **Updates from the live feed:** a v0.1.0 install takes a later version on restart; the M1 test kit's refusals (tampered, wrong key,
   unsigned) still hold with the real configuration.
3. The button's file is the release's file (the sha256 check of step 6), named `Lorren-setup.exe`, delivered as an attachment.
4. The warning steps the website shows match what Windows and Chrome show that day (checked on the fresh account).
5. Uninstall and reinstall both work on that account.

A pre-release goes to the **beta** channel first, to prove the whole path (keys, deploy, live checks) before anything stable exists.

## The builder's calls (settled here; ghost can overrule any)

- **The feed's master folder** is `Programming\lorren-feed\`, outside every repository: it holds the shipped installers, which a
  later build cannot reproduce byte for byte, so it is the one copy that matters and the release script's README says to back it up.
- **The release configuration** is a committed overlay, `src-tauri/tauri.release.conf.json`: the real public key (public by nature)
  and the real endpoint `https://<feed>.pages.dev/app/<channel>/latest.json`. The private key never enters the repository.
- **The first catalog** lists the two models milestone 1 runs, Qwen3-4B and Llama-3.2-3B-Instruct (Q4_K_M, from ungated GGUF
  repositories pinned to a commit), each with its licence; more models are catalog updates, not app releases.
- **A feed the site cannot reach at build time** leaves the line beside the button out (never a stale or invented version); the
  button itself still works, its address being fixed.
- **The beta pre-release** is numbered like `0.1.0-beta.1` (the updater orders pre-releases; the website reads only stable).
- **The fresh-account test** needs a second local Windows account on ghost's PC (a stranger's view: no models in Downloads, no Ollama
  store). Creating one needs his admin consent, so it is in the list below; without it, the test runs under his own account with the
  app's data folder moved aside, which is weaker and says so.

## Order

0. Finish milestone 1: D9 B5-B8 (the measurements with the app's windows, the app's e2e, the group-4 commit, the worktree check).
1. The release script and the feed layout, against the local master folder with the M1 test keys (the deploy step as a dry run), and
   packaging tests on a local feed.
2. The catalog script, Discover, and downloads (§3.7: no account, no Hugging Face page, size and licence and disk room shown before,
   progress with pause and resume, retry with backoff, sha256 as the bytes arrive).
3. The website's half: the line beside the button, the note after the click, the badges checked against the v0.1 build.
4. With ghost's account and keys: the Cloudflare project, the first deploy (a beta pre-release), the live checks.
5. The fresh-account test from the live feed; then v0.1.0 to stable and `VITE_DOWNLOAD_URL` set, and the button is live.

Each step gets its plan (superpowers:writing-plans) and is built test first; the website's half is a plan in `company`.

## Only ghost can do

- A free Cloudflare account with two-factor sign-in, and the feed's name: it becomes `<feed>.pages.dev`, built into every install,
  so it is chosen once and kept (a domain bought later is added to the same project). Then `wrangler login` once, in his browser.
- The two minisign key pairs (app and catalog), made on his PC with the commands the release script's README gives, kept offline
  (USB and a password); he hands over the two public keys.
- His go for the runs that start the engine (the demo's measurements and the fresh-account test): model servers, asked first.
- His OK to create a second local Windows account for the fresh-account test (or to take the weaker test under his own account).

## Not now

A code-signing certificate (it removes the SmartScreen warning; nothing is bought before revenue), a domain, a Mac or Linux build,
an installer for all users (admin), delta updates.
