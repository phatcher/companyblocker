# 2. Aspirational relationships are tagged, not described in comments

Date: 2026-08-31

## Status

Accepted

## Context

The model drew several relationships that the code did not yet make, to show intended architecture. Which ones those were was recorded in a prose comment at the top of `model.dsl` and in `note` blocks in the PlantUML copies. The edges themselves were drawn identically to real ones.

This failed in both directions. A reader of a rendered diagram saw no difference between an edge that existed and one that was merely intended, because the distinction lived only in a comment that does not render. And the comment went stale: it claimed `src/validation` did not call `company_perturbation` long after the perturbation orchestrator and materializer made that dependency real and `tach.toml` declared it. Nothing connected the claim to the thing that would falsify it.

## Decision

A relationship that the architecture intends but the code does not yet make is tagged `Planned` on the relationship itself, and rendered dashed by a `relationship "Planned"` style in `views.dsl`.

The tag is removed at the moment the dependency becomes real. The landing procedure does this in the same step that changes `tach.toml`, and a periodic documentation check compares every `Planned`-tagged edge against `tach.toml` as a backstop for a change that bypasses it.

One category cannot be handled this way. A relationship to a hypothetical external system, such as `downstreamConsumers`, has no in-repo module for `tach.toml` to confirm, so it will never be falsifiable by that check. Those stay documented in prose in `model.dsl`'s header, explicitly scoped as the exception.

## Consequences

The intended-versus-real distinction is visible in every rendered diagram rather than only to someone reading the DSL source.

A stale tag is now detectable mechanically, by comparing tagged edges against `tach.toml`, instead of depending on someone noticing a comment is out of date.

The convention only holds if the flip actually happens at landing time. That is why it is written into the landing procedure rather than left as a convention people are expected to remember, and why the periodic check re-reads it.
