"""Export every diagram in this repo to a PNG under docs/diagrams/png/.

Covers all three diagram families (see docs/diagrams/README.md):

* ``html``        - docs/diagrams/*.html, one PNG per ``<svg>``. Rendered with headless
                    Edge/Chrome, because the SVGs reference CSS custom properties
                    (``var(--accent)``) defined on ``:root`` in the shared
                    ``house-style.css``: a browser is what resolves those. Rasterisers
                    that parse the SVG in isolation render those colours as black.
* ``structurizr`` - docs/structurizr/workspace.dsl, one PNG per view. Exported to
                    PlantUML by the Structurizr CLI, then rendered. Both steps run in
                    Docker, so nothing leaves the machine.
* ``plantuml``    - docs/plantuml/*.puml, the sequence/activity/decision diagrams.

Every run first regenerates the PlantUML and Structurizr palette files from
docs/diagrams/house-style.css (see ``sync_palette.py``).

PNGs carry no padding beyond what each diagram itself defines, and no captions: those
live in the HTML page (for the web) or in whatever document embeds the image.

Usage::

    python tooling/export_diagrams.py                    # everything
    python tooling/export_diagrams.py --only structurizr # one family
    python tooling/export_diagrams.py --filter tier      # names containing "tier"
    python tooling/export_diagrams.py --scale 3          # higher-resolution HTML exports
    python tooling/export_diagrams.py --keys             # include C4 legend/key diagrams
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess  # nosec B404 - dev-tooling script; see nosec B603/B607 at its call sites
import sys
import tempfile
from pathlib import Path

from _tooling_common import repository_root
from sync_palette import sync_palette

REPO_ROOT = repository_root(Path(__file__))
HTML_DIR = REPO_ROOT / "docs" / "diagrams"
PUML_DIR = REPO_ROOT / "docs" / "plantuml"
DSL_DIR = REPO_ROOT / "docs" / "structurizr"
DEFAULT_OUT = HTML_DIR / "png"
# The fonts and palette every HTML page links; inlined into each render page, which
# lives in a temporary directory where the page's relative link would not resolve.
HOUSE_STYLE = HTML_DIR / "house-style.css"

STRUCTURIZR_IMAGE = "structurizr/structurizr"
PLANTUML_IMAGE = "plantuml/plantuml"

BROWSER_CANDIDATES = (
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
)

SVG_RE = re.compile(r"<svg\b[^>]*>.*?</svg>", re.DOTALL)
STYLE_RE = re.compile(r"<style>.*?</style>", re.DOTALL)
# Applied to an `<svg>`'s opening tag only: searched over the whole element, they would
# pick up the first marker's `id` and the first `<rect>`'s size instead of the diagram's.
SVG_TAG_RE = re.compile(r"<svg\b[^>]*>")
ID_RE = re.compile(r'\bid="([^"]+)"')
WIDTH_RE = re.compile(r'\bwidth="([0-9.]+)"')
HEIGHT_RE = re.compile(r'\bheight="([0-9.]+)"')
VIEWBOX_RE = re.compile(r'\bviewBox="[-0-9.]+\s+[-0-9.]+\s+([0-9.]+)\s+([0-9.]+)"')

# Headless Edge can write its screenshot and then never exit, so a render is bounded.
SCREENSHOT_TIMEOUT_SECONDS = 60

PAGE = """<!doctype html>
<html data-theme="light">
<head>
<meta charset="utf-8">
<style>{house_style}</style>
{style}
<style>
  html,body{{margin:0;padding:0;background:var(--surface);}}
  svg{{display:block;}}
</style>
</head>
<body>{svg}</body>
</html>
"""


def kebab(name: str) -> str:
    """SystemContext -> system-context, so exported names match the repo's file style."""
    return re.sub(r"(?<!^)(?=[A-Z])", "-", name).lower()


def run(
    cmd: list[str], what: str, timeout: float | None = None
) -> subprocess.CompletedProcess[str]:
    # Every call site in this module passes a fixed argv (docker/plantuml/browser
    # invocations built from this script's own constants); not shell=True.
    result = subprocess.run(  # nosec B603
        cmd, capture_output=True, text=True, check=False, timeout=timeout
    )
    if result.returncode != 0:
        sys.exit(f"{what} failed:\n{result.stdout[-2000:]}\n{result.stderr[-2000:]}")
    return result


def svg_name_and_size(svg: str, fallback_name: str) -> tuple[str, float, float] | None:
    """The diagram's name and pixel size, read from its opening `<svg>` tag.

    The size comes from `width`/`height`, or from `viewBox` when those are absent; `None`
    when the tag carries neither.
    """
    tag = SVG_TAG_RE.match(svg)
    attrs = tag.group(0) if tag else ""
    ident = ID_RE.search(attrs)
    name = ident.group(1) if ident else fallback_name
    width, height = WIDTH_RE.search(attrs), HEIGHT_RE.search(attrs)
    if width and height:
        return name, float(width.group(1)), float(height.group(1))
    view_box = VIEWBOX_RE.search(attrs)
    if view_box:
        return name, float(view_box.group(1)), float(view_box.group(2))
    return None


def screenshot(cmd: list[str], out_png: Path, what: str) -> None:
    """Run the browser screenshot, accepting a written PNG from a browser that then hangs."""
    try:
        run(cmd, what, timeout=SCREENSHOT_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        if out_png.is_file() and out_png.stat().st_size > 0:
            return
        sys.exit(
            f"{what} timed out after {SCREENSHOT_TIMEOUT_SECONDS}s with no PNG written"
        )


def find_browser() -> str:
    for candidate in BROWSER_CANDIDATES:
        if Path(candidate).is_file():
            return candidate
    found = shutil.which("msedge") or shutil.which("chrome") or shutil.which("chromium")
    if found:
        return found
    sys.exit(
        "No Edge/Chrome found. Install one, or add its path to BROWSER_CANDIDATES."
    )


def require_docker(*images: str) -> None:
    """Check Docker is usable and the images are present, before a confusing failure."""
    if not shutil.which("docker"):
        sys.exit(
            "Docker not found. It is required for the structurizr and plantuml families; "
            "the html family needs only Edge/Chrome, so try --only html."
        )
    # `docker` is resolved via PATH (developer tooling, not attacker-controlled), and
    # both calls below pass fixed argv literals, not shell=True.
    if (
        subprocess.run(  # nosec B603 B607
            ["docker", "info"], capture_output=True, text=True, check=False
        ).returncode
        != 0
    ):
        sys.exit("Docker is installed but not running. Start Docker Desktop and retry.")

    have = subprocess.run(  # nosec B603 B607
        ["docker", "images", "--format", "{{.Repository}}"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.split()
    missing = [i for i in images if i not in have]
    if missing:
        pulls = "\n".join(f"  docker pull {i}" for i in missing)
        sys.exit(f"Missing Docker image(s):\n{pulls}")


def render_puml(puml_dir: Path, out_dir: Path, rename=None, keep=None) -> int:
    """Render every .puml in puml_dir to PNG, then move the results into out_dir."""
    sources = [p for p in sorted(puml_dir.glob("*.puml")) if not p.name.startswith("_")]
    if keep:
        sources = [p for p in sources if keep(p)]
    if not sources:
        return 0

    run(
        ["docker", "run", "--rm", "-v", f"{puml_dir}:/data", PLANTUML_IMAGE, "-tpng"]
        + [f"/data/{p.name}" for p in sources],
        "PlantUML render",
    )

    written = 0
    for source in sources:
        png = source.with_suffix(".png")
        if not png.is_file():
            print(f"  ! {source.name}: no PNG produced")
            continue
        target = out_dir / f"{(rename(source.stem) if rename else source.stem)}.png"
        shutil.move(str(png), target)
        print(f"  {source.name} -> {target.relative_to(REPO_ROOT)}")
        written += 1
    return written


def export_html(out_dir: Path, scale: float, name_filter: str) -> int:
    browser = find_browser()
    sources = sorted(p for p in HTML_DIR.glob("*.html") if name_filter in p.name)
    house_style = HOUSE_STYLE.read_text(encoding="utf-8")
    written = 0

    for source in sources:
        html = source.read_text(encoding="utf-8")
        style = STYLE_RE.search(html)

        for index, svg in enumerate(SVG_RE.findall(html), start=1):
            sized = svg_name_and_size(svg, f"{source.stem}-{index}")
            if sized is None:
                print(
                    f"  ! {source.name}: <svg> {index} has no width/height or viewBox, skipped"
                )
                continue
            name, width, height = sized

            out_png = out_dir / f"{name}.png"
            page = PAGE.format(
                house_style=house_style,
                style=style.group(0) if style else "",
                svg=svg,
            )
            # A browser killed on timeout can leave its profile locked on Windows.
            with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
                src = Path(tmp) / "page.html"
                src.write_text(page, encoding="utf-8")
                screenshot(
                    [
                        browser,
                        "--headless=new",
                        "--disable-gpu",
                        "--hide-scrollbars",
                        f"--user-data-dir={Path(tmp) / 'profile'}",
                        f"--force-device-scale-factor={scale}",
                        f"--window-size={round(width)},{round(height)}",
                        "--virtual-time-budget=4000",  # let webfonts load before the shot
                        f"--screenshot={out_png}",
                        src.as_uri(),
                    ],
                    out_png,
                    f"Screenshot of {name}",
                )
            print(f"  {source.name} -> {out_png.relative_to(REPO_ROOT)}")
            written += 1
    return written


def export_structurizr(out_dir: Path, name_filter: str, include_keys: bool) -> int:
    require_docker(STRUCTURIZR_IMAGE, PLANTUML_IMAGE)
    workspace = DSL_DIR / "workspace.dsl"
    if not workspace.is_file():
        print(f"  ! {workspace} not found, skipped")
        return 0

    with tempfile.TemporaryDirectory() as tmp:
        staging = Path(tmp)
        run(
            [
                "docker",
                "run",
                "--rm",
                # docs/ rather than docs/structurizr/: workspace.dsl pulls in
                # ../architecture and ../decisions, which must be inside the mount.
                "-v",
                f"{DSL_DIR.parent}:/ws:ro",
                "-v",
                f"{staging}:/out",
                STRUCTURIZR_IMAGE,
                "export",
                "-w",
                "/ws/structurizr/workspace.dsl",
                "-f",
                "plantuml",
                "-o",
                "/out",
            ],
            "Structurizr export",
        )

        def keep(path: Path) -> bool:
            if not include_keys and path.stem.endswith("-key"):
                return False  # the legend, not a view
            return name_filter in path.stem

        # structurizr-SystemContext -> c4-system-context
        def rename(stem: str) -> str:
            return "c4-" + kebab(stem.removeprefix("structurizr-"))

        return render_puml(staging, out_dir, rename=rename, keep=keep)


def export_plantuml(out_dir: Path, name_filter: str) -> int:
    require_docker(PLANTUML_IMAGE)
    with tempfile.TemporaryDirectory() as tmp:
        staging = Path(tmp)
        for source in PUML_DIR.glob("*.puml"):
            shutil.copy2(source, staging / source.name)  # includes _style.puml
        return render_puml(staging, out_dir, keep=lambda p: name_filter in p.stem)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--only",
        choices=("html", "structurizr", "plantuml"),
        help="export just one family",
    )
    parser.add_argument(
        "--filter", default="", help="only diagrams whose name contains this"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help=f"output dir (default: {DEFAULT_OUT})",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=2.0,
        help="device scale for HTML exports (default: 2)",
    )
    parser.add_argument(
        "--keys", action="store_true", help="also export C4 legend/key diagrams"
    )
    args = parser.parse_args()

    out_dir = args.out if args.out.is_absolute() else REPO_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    # PlantUML and Structurizr cannot read house-style.css, so their palette files are
    # regenerated from it first: a render never uses a stale palette.
    for path in sync_palette():
        print(f"palette -> {path.relative_to(REPO_ROOT)}")

    families = [args.only] if args.only else ["html", "structurizr", "plantuml"]
    total = 0
    for family in families:
        print(f"[{family}]")
        if family == "html":
            total += export_html(out_dir, args.scale, args.filter)
        elif family == "structurizr":
            total += export_structurizr(out_dir, args.filter, args.keys)
        else:
            total += export_plantuml(out_dir, args.filter)

    print(f"\n{total} PNG(s) in {out_dir.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
