"""Command-line settings this package's own scripts read, declared as plain data.

A script reads these to build its flags, name each default in its help, and
report which settings applied to the run a `--dry-run` reported. Nothing here
builds a parser: the package stays a library, and the declaration shape is a
documented convention rather than a type shared with other packages. See
`packages/company_tokenize/src/company_tokenize/_cli_helper.py` for another
package's version of the same convention.
"""

from __future__ import annotations

SETTINGS: tuple[dict[str, object], ...] = (
    {
        "name": "iso20275_countries",
        "type": "str",
        "default": None,
        "help": (
            "Comma-separated country codes scoping the ISO 20275 mapping "
            "proposal and audit; unset covers every country the rules table "
            "already has entries for."
        ),
    },
    {
        "name": "short_name_layer_seed",
        "type": "int",
        "default": 0,
        "help": (
            "Seed for the deterministic entity-level train/test/validate "
            "split each short-name layer is measured against."
        ),
    },
    {
        "name": "short_name_layer_german_decompounding",
        "type": "bool",
        "default": False,
        "help": (
            "Inject the wordfreq-based German compound splitter into the "
            "type_prefix_acronym_qualifier layer for de-language pairs, to "
            "measure its lift."
        ),
    },
)
"""Every setting this package's own scripts (not a caller's) take."""
