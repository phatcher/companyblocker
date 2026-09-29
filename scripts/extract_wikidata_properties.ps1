<#
.SYNOPSIS
Build a wdgrep property graph from the raw Wikidata dump.

.DESCRIPTION
Streams the gzipped dump through `wdgrep build-graph`, writing one graph index
per acquire date. Replaces the earlier company_grep.ps1/company_wikidata_index.ps1
pair, which wrote the same output by two different routes.

Reuse is explicit rather than silent: an existing index is kept and reported,
and rebuilt only with -Force. If the existing index was built for different
properties than requested, that is a warning, not a silent reuse -- the failure
mode this avoids is a successful-looking run that consumed a stale artifact
built for a different property set.

This builds the raw P279 graph, which is the upstream half of the company-type
subclass closure the Wikidata extractors read as p279.json (49,176
`{"subclass": "<uri>"}` entries). The step that reduces this graph to that array
does not exist in this repo yet, so the two files are different formats and this
script must not be pointed at p279.json: writing graph output there would
replace a working extractor input with an incompatible shape.

.PARAMETER AcquireDate
Acquire-date directory under data/wikidata/acquire to read and write.

.PARAMETER Properties
Comma-separated Wikidata property IDs to build the graph over.

.PARAMETER Force
Rebuild even when an index already exists.

.EXAMPLE
pwsh -File scripts/extract_wikidata_properties.ps1 -Properties P279

.EXAMPLE
pwsh -File scripts/extract_wikidata_properties.ps1 -AcquireDate 2026-07-16 -Force
#>
param(
	[string]$AcquireDate = "2026-07-16",
	[string]$Properties = "P279",
	[switch]$Force
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$acquirePath = Join-Path $repoRoot "data/wikidata/acquire/$AcquireDate"
$inputPath = Join-Path $acquirePath "wikidata-all.json.gz"
$outputPath = Join-Path $acquirePath "graph.ndjson"
$metaPath = "$outputPath.meta.json"
$sevenZip = "C:\Program Files\7-Zip\7z.exe"

if (-not (Test-Path $sevenZip)) {
	throw "7-Zip executable not found at: $sevenZip"
}

if (-not (Get-Command wdgrep -ErrorAction SilentlyContinue)) {
	throw "wdgrep not found on PATH. Install it with: cargo install wdgrep"
}

if (-not (Test-Path $inputPath)) {
	throw "Wikidata input file not found: $inputPath"
}

# Reuse is a decision, not a default: say what is being reused and what it was
# built for, so a stale index can be told apart from an intended resume.
if ((Test-Path $outputPath) -and (-not $Force)) {
	$builtFor = $null
	if (Test-Path $metaPath) {
		$builtFor = (Get-Content $metaPath -Raw | ConvertFrom-Json).properties
	}

	if ($null -eq $builtFor) {
		Write-Warning "Existing index $outputPath has no companion $([System.IO.Path]::GetFileName($metaPath)), so the properties it was built for are unknown. Re-run with -Force to rebuild for '$Properties'."
	}
	elseif ($builtFor -ne $Properties) {
		Write-Warning "Existing index $outputPath was built for properties '$builtFor', not the requested '$Properties'. Re-run with -Force to rebuild."
	}
	else {
		Write-Host "[wikidata-properties] reusing existing index for '$Properties': $outputPath"
	}

	Write-Host "[wikidata-properties] nothing to do (use -Force to rebuild)"
	exit 0
}

Write-Host "[wikidata-properties] building index for '$Properties'"
Write-Host "[wikidata-properties] input:  $inputPath"
Write-Host "[wikidata-properties] output: $outputPath"

# Single streaming pass: 7-Zip decompresses to stdout and wdgrep consumes it
# directly, so the full dump is never materialized as an intermediate file.
& $sevenZip x $inputPath -so | wdgrep build-graph --properties $Properties > $outputPath

if ($LASTEXITCODE -ne 0) {
	throw "wdgrep build-graph failed with exit code $LASTEXITCODE"
}

[ordered]@{
	properties  = $Properties
	acquireDate = $AcquireDate
	source      = [System.IO.Path]::GetFileName($inputPath)
	builtUtc    = (Get-Date).ToUniversalTime().ToString("o")
} | ConvertTo-Json | Set-Content -Path $metaPath -Encoding utf8

Write-Host "[wikidata-properties] wrote $outputPath"
