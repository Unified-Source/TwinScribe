# Specification: the site

Read `CONVENTIONS.md` first, then `icon.md` for the mark and the wordmark. Deliver `site/`
(the page, its stylesheet and its script), `tools/build_site.py` with `tests/test_site.py`,
the workflow `.github/workflows/site.yml`, and the site's address in the README and in
`pyproject.toml`.

## 1. One page

The site is one page at the project's own address, twinscribe.app. It says what the README
says, shorter, for a reader who has not opened the repository: the mark, the name and the
tagline; the claim in one sentence; the two-engine design with the figures it was measured
by; a transcript as the window shows it; the outputs per recording; the window and the
verification screen; the three ways to install; what the tool is not yet; the components and
their licences. Every sentence is drawn from the README or the release notes and keeps their
register: impersonal, British spelling, ASCII hyphens. The page names no tooling, and no one
outside the repository beyond the copyright holder named in `NOTICE`.

The pictures are the documentation's own: the live recording as the MP4 beside the claim,
the window playing a transcribed recording, and the verification screen. The transcript
excerpt is the first chapter of the public-domain audiobook the README's stills show, with
the lines as the window laid them out, including a silence marked and a mark for missed
speech; the two words the second engine heard there are the two the book has and the
published line lacks, so the excerpt is a checkable specimen rather than an illustration.

## 2. Download

The download button carries the Releases page as its address in the markup, so the page reads
right with no script. The script asks the repository's releases for the newest one and, when
it answers, points the button at `twinscribe-win64.zip` and states the version, the size and
the digest beside it; without an answer nothing changes. That request is the one thing on the
page that reaches another service: no font, script or style is loaded from elsewhere, and the
page carries no analytics.

## 3. Colours and type

The page follows the mark: the banner's dark ground and bright blue by default, the window's
light palette when the reader's system asks for light, each defined as a full set of tokens
so that every colour resolves in both. "Twin" is set in the text colour and "Scribe" in the
blue, as the window's top bar sets them, and the tagline in spaced capitals under the name as
the banner has it. The wordmark's family is the one `app_icon.py` names for the rendered
files; a machine without it renders its bold sans-serif, as the program does. Timestamps,
file names and keys are set in the machine's monospace face. The two engines are drawn as the
mark draws them, two facing panels, the published one light and the checking one blue, with
the dot between; a review mark is shown as the window shows it, a strip between the lines.

## 4. Assembly and publication

`tools/build_site.py` copies `site/` into an output folder with the pictures, the recording
and the icon the page shows, taken from `docs/images/`, `docs/media/` and `assets/`, so that
no picture is stored twice. It then checks that every local reference in the page resolves in
the assembled folder and that the site's text files hold ASCII only, and fails on either. The
workflow runs it on every push to `main` that touches those files and publishes the folder
through GitHub Pages; the site's address is a setting of the repository and a record at the
domain's registrar, both outside the repository.

## 5. Tests

The reference parser on a hand-written snippet keeps the relative paths and drops addresses
elsewhere, fragments and inline data. The assembled folder holds the page and the icon and
resolves every reference; a stale folder is replaced; a missing reference is reported with its
page. The site's text files are ASCII with no dash but the hyphen, and a file that is not is
found with its line. The page carries the name, the wordmark's two spans and the tagline, and
no generator line.
