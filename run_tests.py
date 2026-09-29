import pytest


def main() -> int:
    return pytest.main(
        [
            "-vv",
            "src/tests/acquisition/test_acquisition_pipeline.py",
            "src/tests/acquisition/test_registry.py",
            "src/tests/acquisition/test_extractors.py",
            "src/tests/acquisition/test_chunking.py",
            "src/tests/acquisition/test_wikidata_runtime.py",
            "src/tests/acquisition/test_downloader_wikidata.py",
            "src/tests/cleanser/test_orchestrate.py",
            "src/tests/cleanser/test_configure.py",
            "src/tests/cleanser/test_query_plan.py",
            "src/tests/cleanser/test_smoke.py",
            "src/tests/cleanser/test_cleanser_ops.py",
            "src/tests/cleanser/test_cleanser_extract.py",
            "src/tests/cleanser/test_chunk_naming.py",
            "src/tests/acquisition/test_private_analysis.py",
            "src/tests/acquisition/test_acquisition_tokenizer_ops.py",
            "src/tests/acquisition/test_acquisition_tokenizer_promote.py",
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
