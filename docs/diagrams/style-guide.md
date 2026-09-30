# Diagram style

Shared visual identity for every diagram family in this repo: the C4/Structurizr structural model, the UML/PlantUML behavioural diagrams, and the HTML/SVG explainer artifacts. Reuse these tokens rather than deriving a new palette or type pairing per diagram or per family, so a reader moving between diagrams is not re-learning a visual language. See [`README.md`](README.md) for which family to use when.

The palette and type below are medium-agnostic and apply everywhere. The layout conventions further down are HTML/SVG-specific and apply only to the explainer artifacts.

Applying the identity per toolchain:

- **Structurizr (C4)**: `views.dsl`'s `styles { }` block, on `element` and `relationship` selectors, names palette colours as constants (`${ACCENT}`) from the generated `_palette.dsl`.
- **PlantUML (UML)**: `docs/plantuml/_style.puml` sets skinparams from the variables (`$accent`, `$font_body`) in the generated `_palette.puml`, and is pulled into each diagram with `!include _style.puml`.
- **HTML/SVG (explainers)**: [`house-style.css`](house-style.css), which every page links, holds the fonts, the palette as CSS custom properties, the page skeleton and every component. A page carries no `<style>` of its own, so one edit there restyles every explainer.

Structurizr and PlantUML each render one fixed theme, so both use the light palette only. The dark palette applies to HTML artifacts, which follow the viewer's theme.

This is a house style for this project's own diagrams, not a rule for every artifact. A landing page, a game, or a one-off tool for someone outside this project gets a bespoke treatment built for its own subject.

`resilience-ladder.html` (this folder) is a published explainer built with these conventions. A new page links `house-style.css` and builds from the skeleton its header comment lists (`.page`, `header.intro`, `.legend`, `.section`, `.steps`/`.step`, `figure` with `.diagram-wrap`, `.asides`); a component a page needs that the file lacks is added there, never to the page. Copy the card, cylinder and connector patterns from `resilience-ladder.html`'s SVG rather than re-deriving them by hand. An SVG that sets no fonts by attribute takes `class="diagram"` for the default text style. Its content, the three-step `gleif`/`gb` narrative with its specific cards and captions, is that diagram's own subject matter: a new diagram reuses the tokens, card shape and connector conventions, and invents its own structure and however many steps or panels its subject needs.

## Palette

The blue accent is also the C4 Container colour, so the structural diagrams and everything else read as one project. The rest of the C4 element ramp, a scale of the same blue (`#0B3C73` person, `#1A4F8B` software system, `#7FA3CC` component), stays Structurizr's own, in `views.dsl`: restyle a diagram to match the ramp, not the ramp to match a diagram.

The page and cards are white. Around them sits a cool blue-grey neutral, tied to the accent by hue rather than a default mid-grey, and two semantic accents: blue for a mechanism or lineage relationship, something that flows or points back to where it came from, and amber for a cross-system truth or verification signal. Amber is the one warm note, since the C4 ramp is entirely cool and a second cool accent would not read as a distinct meaning. A diagram needing a third semantic colour extends this set once agreed here, rather than reaching for an arbitrary hue.

The values, light and dark, are in [`house-style.css`](house-style.css). Its dark set is defined under `@media (prefers-color-scheme: dark)` guarded by `:root:not([data-theme="light"])`, and again under `:root[data-theme="dark"]`, so an explicit toggle wins in both directions. Neither Structurizr nor PlantUML reads CSS, so `tooling/sync_palette.py` writes the light values and the fonts into `docs/structurizr/_palette.dsl` and `docs/plantuml/_palette.puml`. `tooling/export_diagrams.py` runs it before every export; both files are generated, so edit `house-style.css`, never them.

## Type

- Display and heading: Archivo (700/800/900), a technical grotesque with enough weight to carry a title without reading as a landing-page hero.
- Body: Source Sans 3 (400/500/600), plain and legible for running text.
- Mono: IBM Plex Mono (400/500/600), reserved for identifiers, URIs, field names and code-shaped values specifically, not for technical text generally. That restraint is what makes a `system_uri` value read as a value rather than a label.

`house-style.css` loads all three from Google Fonts and names them `--font-display`, `--font-body` and `--font-mono`. Pages refer to those names, in CSS and in SVG `font-family` attributes alike, rather than to a font directly.

## Layout conventions (HTML/SVG explainers only)

These are medium-specific: they describe how to build an HTML/SVG explainer artifact and do not transfer to Structurizr or PlantUML, which have their own element vocabularies.

- A stage or record is a card: an SVG `rect` or a bordered `div`, `rx` 8-10, `var(--surface)` fill, `var(--line)` stroke. Reserve `var(--accent)` stroke at 2px weight for the one row anchoring the diagram, the origin or root record.
- A pipeline reads left to right as labelled column groups: small caps, `var(--muted)`, `IBM Plex Mono`, letter-spacing around 0.14em, with a thin rule underneath. A muted `›` glyph in the gap between groups marks macro stage progression, kept visually distinct from any field-level connector.
- A relationship between two specific fields gets its own connector, never a shared mesh connecting everything to everything:
  - Lineage or mechanism, a value that names its own origin: solid `var(--accent)` line, routed as an orthogonal elbow (a native SVG `path`, no curve library), arrowhead as a small filled `polygon`, labelled once with a rotated mono label (`transform="rotate(-90 …)"`) along the vertical run.
  - Cross-system truth, or a value copied through unchanged: dashed `var(--amber)` line, straight, with the same value shown at both ends so the unchanged claim is visible without reading a label.
- A whole dataset, as opposed to a single row or record, is a small cylinder: two ellipses (`rx` ~38, `ry` ~8) joined by two vertical lines, `var(--surface-2)` fill, `var(--line)` stroke. Reserve it for a whole materialized artifact entering the diagram, not for something that is one row.
- `figure` plus `figcaption`: the SVG carries the mechanism and the caption carries the one sentence of nuance a picture cannot show, such as an edge case, a multi-hop resolution or a scoring rule. A caption needing more than a sentence or two means the diagram is missing something it should be showing.
- A callout or aside, for a design rationale, an open question or a caveat, is a bordered block on `var(--surface-2)`: `border-left: 3px solid var(--accent)` for a settled rationale, `var(--line)` for something still open. Reuse those two states rather than inventing a border colour per note.

## When this applies

The palette and type apply to every diagram in this repo, across all three families. The HTML/SVG layout conventions apply to explainer artifacts about this project's own mechanisms. Neither is a rule for artifacts built for an audience outside this project.
