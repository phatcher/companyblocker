!identifiers hierarchical

# Implementation status: genuinely-planned (not-yet-real) relationships
# are tagged "Planned" on the relationship itself below and rendered
# dashed via views.dsl's styles block, rather than tracked only in this
# comment. Flip the tag (drop "Planned", restyle as a normal edge) at the
# change that actually wires the dependency in and updates tach.toml.
#
# Two relationships can't be tagged this way, since downstreamConsumers
# is a hypothetical external system with no in-repo module tach.toml can
# confirm a dependency against: "pipeline.packages.vectorizePackage ->
# downstreamConsumers" and "pipeline.packages.classifyPackage ->
# downstreamConsumers" (no trained artifact is currently persisted or
# exported for either) stay documented here in prose instead.

dataEngineer = person "Data Engineer" "Runs acquisition and processing scripts, monitors outputs, and curates onboarding changes."
dataAnalyst = person "Data Analyst" "Uses notebooks and pipeline outputs for analysis and iterative data quality review."

upstreamRegistries = softwareSystem "Upstream Company Registries" "Country and global publishers (for example GB Companies House, INSEE, CRO, GLEIF)." "External"
downstreamConsumers = softwareSystem "Downstream Analytics/ML Workloads" "Jobs and notebooks that consume cleansed and tokenized outputs." "External"
externalConsumers = softwareSystem "External Python Consumers" "Other Python projects importing the packages directly, outside this platform's own pipeline." "External"

pipeline = softwareSystem "CompanyBlocker" "An exploratory research platform for blocking algorithms on company names: acquiring, transforming, cleansing, tokenizing, vectorizing/blocking, training, and validating company data, built on independently reusable packages." {
    # The prose architecture docs and decision records render alongside the diagrams in
    # Structurizr Lite. Both live outside this directory and are pulled in by reference
    # rather than copied: docs/architecture/*.md is the canonical home for the prose, and
    # duplicating it here would recreate exactly the drift this model exists to avoid.
    # Lite therefore mounts docs/ (see .structurizr/docker-compose.yaml).
    !docs ../architecture
    !adrs ../decisions

    cli = container "CLI Entry Points" "Standalone top-level scripts for each stage and for end-to-end runs." "Python"
    orchestration = container "Acquisition/Processing Orchestration" "Executes system-specific acquisition and post-acquisition stages, owning this repo's data plumbing around calls into Packages." "Python" {
        downloader = component "Downloaders & Acquire Pipeline" "Acquire pipeline that fetches and unpacks bulk/API source data into run-date acquire folders and prepared projections." "Python"
        sharding = component "Sharding" "Coordinates source-to-shard orchestration, including resource dispatch, freshness-aware batch processing, and system-specific sharding fallbacks." "Python"
        shardingBatchProcessor = component "Shard Batch Processor" "Processes prepared resources through transform/filter/materialize/write and delegates handler-specific custom logic." "Python"
        shardingHandlers = component "Shard System Handlers" "Resolves per-system sharding handlers and encapsulates adapter-specific custom logic for DBpedia, GLEIF, Wikidata, Offeneregister, and GB." "Python"
        wikidataProjection = component "Wikidata Projection" "Projects the Wikidata dump through the wikisieve or native Python engine and writes the run manifest." "Python (downloader_wikidata.py, wikidata_runtime.py)"
        canonicalization = component "Canonicalization" "Maps sharded source records into an OpenCorporates-aligned canonical schema." "Python"
        cleansing = component "Cleansing Orchestrator" "Resolves canonical inputs and invokes the cleanse engine with system-specific mappings." "Python"
        match = component "Match Stage" "Labels a source system's canonical rows with their cross-system match_uri against a target system." "Python (match_ops.py)"
        tokenization = component "Tokenizer Ops" "Samples tokenizer training corpora from cleansed names and defines the tokenized output columns." "Python (tokenizer_ops.py)"
        registry = component "System Registry & Catalog" "Loads catalog plans, system metadata, and per-system behavior flags used by all stages." "Python/JSON"
    }
    rustExtractors = container "Wikidata Rust Extractors" "In-repo Rust binary (wikisieve) that projects the compressed Wikidata dump." "Rust (src/rust)"
    analysis = container "Analysis Workflows" "Analysis and reporting workflow for token quality, benchmarks, and diagnostics." "Python" {
        analysisCli = component "Analysis Entry Points" "Top-level scripts that load tokenized data, run benchmarks, and generate analysis reports." "Python"
        analysisLibrary = component "Analysis Library" "Reusable analysis modules under src/analysis for diagnostics and metrics." "Python"
    }
    training = container "Training Workflows" "Tokenizer training orchestration with workload management, runtime scheduling, optimization, and promotion." "Python (src/training)" {
        trainCli = component "Training Entry Points" "Top-level scripts for tokenizer training, hyperparameter optimization, and candidate archiving." "Python"
        trainLibrary = component "Training Library" "Reusable training modules under src/training for workload scheduling, optimization execution, and promotion." "Python"
    }
    blocking = container "Blocking Workflow" "Two-dataset candidate generation: load, generate candidates, cluster, and score against ground truth." "Python (src/blocking)" {
        blockingContracts = component "Blocking Contracts" "Run contract and two-dataset input model: descriptor resolution, strategy config, run-config validation." "Python (contracts.py)"
        blockingLoader = component "Blocking Loader" "Resolves a system's matched layer (source) or latest canonical snapshot (target) and discovers its countries." "Python (loader.py)"
        blockingNameTransform = component "Blocking Name Transform" "Derives both sides' name forms under the run's cleanse profile and selects the scored column." "Python (name_transform.py)"
        blockingTruth = component "Blocking Truth" "Source-side ground truth: matched-layer match_uri, or a named column on the rows." "Python (truth.py)"
        blockingWorkflow = component "Blocking Workflow Orchestration" "execute_blocking_run(): builds the text view, generates and prunes candidates, clusters, and scores." "Python (workflow.py)"
        blockingRunLayout = component "Blocking Run Layout" "Names each run's location under artifacts/blocking/ and every artefact inside it." "Python (run_layout.py)"
        blockingReporting = component "Blocking Reporting" "Serializes a blocking run result to validated Parquet/JSON artefacts." "Python (reporting.py)"
        blockingStoredRuns = component "Blocking Stored Runs" "Reads finished runs back from disk as comparison entries." "Python (stored_runs.py)"
        blockingComparison = component "Blocking Strategy Comparison" "Reshapes several completed runs into one label/stage/country-keyed comparison frame." "Python (comparison.py)"
        blockingAudit = component "Blocking Pair Audit" "Audits a finished run's missed truth pairs against the exact scan at the run's own settings." "Python (audit.py)"
        blockingInspection = component "Blocking Inspection" "Reshapes a run result into plot-ready frames for notebook inspection." "Python (inspection.py)"
    }
    validation = container "Validation Workflows" "Model and data validation framework for clustering, entity resolution, and ML model evaluation with configurable test matrices." "Python (src/validation)" {
        validationCli = component "Validation Entry Points" "Top-level scripts for running validation suites against trained models and prepared datasets." "Python"
        validationLibrary = component "Validation Library" "Reusable validation modules under src/validation for test execution and reporting." "Python"
    }
    packages = container "Packages" "Standalone, independently publishable packages that do the cleansing, tokenization, vectorization, classification, perturbation, and resolution work." "Python (packages/*)" {
        cleanseRules = component "Company Cleanse Engine" "Deterministic company-name cleansing and company-type resolution engine, applying ISO-20275-grounded rules." "Python (packages/company_cleanse)"
        tokenizePackage = component "Company Tokenize Package" "Standalone tokenization engine for inference-time company name tokenization with WordPiece/SentencePiece models." "Python (packages/company_tokenize)"
        vectorizePackage = component "Company Vectorize Package" "Feature generation and vectorization engine for company matching using embeddings and similarity scoring strategies." "Python (packages/company_vectorize)"
        classifyPackage = component "Company Classify Package" "Company-name validity and pair classifiers, with their training, evaluation and inference." "Python (packages/company_classify)"
        perturbationPackage = component "Company Perturbation Package" "Deterministic name-mutation operator engine producing perturbed name variants for robustness evaluation." "Python (packages/company_perturbation)"
        resolversPackage = component "Company Resolvers Package" "Resolver contract composing a candidate source, a pair scorer, and a decision policy." "Python (packages/company_resolvers)" {
            # Nothing in the repo composes a resolver yet, so it has no relationship to draw.
            properties {
                "structurizr.inspection.model.element.disconnected" "ignore"
            }
        }
    }
    workspaceLib = container "Workspace Layout" "Owns every data-layer and artifact path rule, references, production records, the tokenizer pointer, and the name-forms store." "Python (src/workspace)"
    workspaceStorage = container "Workspace Storage" "Repository-local data/ layers and perturbed datasets, artifacts/ (analysis, blocking, validation, tokenizers, store, pretrained_vectors, perf), config/tokenizers.json, and mlflow.db." "Filesystem (Parquet/ZIP/JSON/TXT/SQLite)"
    notebooks = container "Exploration Notebooks" "Analysis notebooks for token, match, pair-outcome, residual, and perturbation exploration." "Jupyter Notebook"
}

dataEngineer -> pipeline.cli "Runs pipeline stages and full processing commands" "CLI"
dataAnalyst -> pipeline.notebooks "Explores analysis outputs, blocking residuals, tokenizers, and perturbations" "Jupyter Notebook"
dataAnalyst -> pipeline.analysis "Runs analysis reports and performance diagnostics" "CLI"

pipeline.cli -> pipeline.orchestration "Invokes stage orchestration and end-to-end pipeline" "Python"
pipeline.cli -> pipeline.training "Invokes model training workflows" "Python"
pipeline.cli -> pipeline.validation "Invokes model and data validation" "Python"
pipeline.cli -> pipeline.blocking "Invokes the two-dataset blocking workflow (scripts/run_blocking.py)" "Python"
pipeline.cli -> pipeline.workspaceLib "Resolves roots, run inputs, and the promoted tokenizer" "Python"
pipeline.cli -> pipeline.workspaceStorage "Reads/writes run inputs, outputs, and artifacts" "Filesystem (Parquet/JSON)"
pipeline.cli -> pipeline.packages.tokenizePackage "Tokenizes cleansed names in the tokenize stage and derives TF-IDF noise words" "Python"
pipeline.cli -> pipeline.packages.vectorizePackage "Resolves clustering strategies for blocking runs and comparisons" "Python"
pipeline.cli -> pipeline.packages.classifyPackage "Measures pair-classifier ceilings and builds promotion reports" "Python"
pipeline.cli -> pipeline.packages.cleanseRules "Promotes validated noise-word candidates into company_cleanse's packaged defaults" "Python"
pipeline.notebooks -> pipeline.analysis.analysisLibrary "Explores analysis metrics and pictures interactively" "Python"
pipeline.notebooks -> pipeline.blocking.blockingInspection "Reads run frames for residual-pair inspection" "Python"
pipeline.notebooks -> pipeline.packages.perturbationPackage "Explores perturbation operators and profiles" "Python"
pipeline.notebooks -> pipeline.packages.tokenizePackage "Inspects trained tokenizers" "Python"
pipeline.notebooks -> pipeline.workspaceLib "Resolves artifact and run locations" "Python"
pipeline.notebooks -> pipeline.workspaceStorage "Reads intermediate and final artifacts" "Filesystem (Parquet/JSON)"
pipeline.analysis.analysisCli -> pipeline.analysis.analysisLibrary "Uses shared analysis metrics and loaders" "Python"
pipeline.training.trainCli -> pipeline.training.trainLibrary "Uses shared training orchestration and optimization" "Python"
pipeline.validation.validationCli -> pipeline.validation.validationLibrary "Uses shared validation and test infrastructure" "Python"

pipeline.orchestration -> upstreamRegistries "Downloads source datasets via HTTP/API/bulk files" "HTTP/API/Bulk files"
pipeline.orchestration -> pipeline.packages.cleanseRules "Invokes company_cleanse's engine to normalize names and resolve company type" "Python"
pipeline.orchestration -> pipeline.packages.tokenizePackage "Normalizes tokenizer training names via company_tokenize" "Python"
pipeline.orchestration -> pipeline.workspaceLib "Resolves data-layer directories and snapshots" "Python"
pipeline.orchestration -> pipeline.workspaceStorage "Persists acquire, prepare, source, canonical, cleansed, and matched layers and training corpora" "Filesystem (Parquet/JSON)"
pipeline.rustExtractors -> pipeline.workspaceStorage "Reads the compressed Wikidata dump and writes projection chunks" "Filesystem"
pipeline.analysis -> pipeline.workspaceLib "Resolves report and artifact locations" "Python"
pipeline.analysis -> pipeline.workspaceStorage "Reads tokenized data and writes analysis artifacts" "Filesystem (Parquet/JSON)"
pipeline.analysis -> pipeline.validation "Reads run-metric schemas and truth-bucket names" "Python"
pipeline.analysis.analysisLibrary -> pipeline.validation.validationLibrary "Reads run-metric schemas, recall-curve area, and truth-bucket names" "Python"
pipeline.analysis.analysisLibrary -> pipeline.packages.cleanseRules "Reads company-type rules and noise-word data for diagnostics" "Python"
pipeline.analysis.analysisLibrary -> pipeline.packages.tokenizePackage "Uses company_tokenize's TF-IDF rare-token selection for corpus rarity/Zipf diagnostics" "Python"
pipeline.analysis.analysisLibrary -> pipeline.packages.classifyPackage "Reads company_classify's MetricBundle for promotion reports" "Python"
pipeline.training -> pipeline.workspaceLib "Stores candidates as records and moves the promoted pointer" "Python"
pipeline.training -> pipeline.workspaceStorage "Reads cleansed names, writes artifacts/tokenizers, and updates config/tokenizers.json" "Filesystem (Parquet/JSON)"
pipeline.training -> pipeline.orchestration "Samples training corpora via tokenizer_ops" "Python"
pipeline.training -> pipeline.analysis "Reads Zipf summaries via token_zipf" "Python"
pipeline.training.trainLibrary -> pipeline.orchestration.tokenization "Samples training corpora" "Python"
pipeline.training.trainLibrary -> pipeline.analysis.analysisLibrary "Reads the latest Zipf summary for corpus reports" "Python"
pipeline.training.trainLibrary -> pipeline.packages.tokenizePackage "Trains and optimizes tokenizers and writes candidate directories via company_tokenize" "Python"
pipeline.training.trainLibrary -> pipeline.packages.vectorizePackage "Trains and optimizes vectorization models" "Python" {
    tags "Planned"
}
pipeline.validation -> pipeline.workspaceLib "Resolves datasets, references, and the target-index cache location" "Python"
pipeline.validation -> pipeline.workspaceStorage "Reads prepared datasets and trained models, writes validation reports" "Filesystem (Parquet/JSON)"
pipeline.validation -> pipeline.orchestration "Looks up system plans, company-type mappings, and country counts" "Python"
pipeline.validation.validationLibrary -> pipeline.orchestration.registry "Looks up system plans and company-type mappings" "Python"
pipeline.validation.validationLibrary -> pipeline.packages.classifyPackage "Evaluates classifier and blocking models" "Python" {
    tags "Planned"
}
pipeline.validation.validationLibrary -> pipeline.packages.vectorizePackage "Evaluates vectorization quality metrics" "Python"
pipeline.validation.validationLibrary -> pipeline.packages.perturbationPackage "Invokes company_perturbation's operators for perturbation profiles" "Python"
pipeline.validation.validationLibrary -> pipeline.packages.cleanseRules "Invokes cleanse_lazyframe() for name forms, GLEIF name-variant collapse, and perturbed datasets" "Python"
pipeline.validation.validationLibrary -> pipeline.packages.tokenizePackage "Tokenizes names via company_tokenize during validation runs" "Python"
pipeline.blocking -> pipeline.workspaceLib "Resolves layers, run references, the promoted tokenizer, and cached name forms" "Python"
pipeline.blocking -> pipeline.workspaceStorage "Reads matched source and canonical target layers, reads/writes artifacts/store, writes artifacts/blocking" "Filesystem (Parquet/JSON)"
pipeline.blocking -> pipeline.packages.tokenizePackage "Tokenizes text views for token-list representations" "Python"
pipeline.blocking -> pipeline.packages.cleanseRules "Reads effective noise words for the target-index key" "Python"
pipeline.blocking -> pipeline.packages.vectorizePackage "Delegates candidate-generation and similarity strategy logic" "Python"
pipeline.blocking -> pipeline.validation "Derives name forms, caches target indexes, and scores candidates via compute_pair_truth_eval()" "Python"
downstreamConsumers -> pipeline.workspaceStorage "Consumes cleansed/tokenized datasets and tokenizer artifacts" "Filesystem (Parquet/JSON)"
downstreamConsumers -> pipeline.packages.vectorizePackage "Uses trained vectorization models" "Python"
downstreamConsumers -> pipeline.packages.classifyPackage "Uses trained classifier models" "Python"
pipeline.packages.classifyPackage -> pipeline.packages.perturbationPackage "Generates perturbed training pairs via company_perturbation operators" "Python"
pipeline.packages.perturbationPackage -> pipeline.packages.cleanseRules "Reads company_cleanse's homoglyph, ISO-20275 company-type, and corpus noise-word data" "Python"
externalConsumers -> pipeline.packages.cleanseRules "Imports company_cleanse directly as a standalone library" "Python package import"
externalConsumers -> pipeline.packages.perturbationPackage "Imports company_perturbation directly as a standalone library" "Python package import"
pipeline.workspaceLib -> pipeline.workspaceStorage "Owns the layout of data/, artifacts/, and config/tokenizers.json" "Filesystem (Parquet/JSON)"

pipeline.orchestration.downloader -> upstreamRegistries "Fetches source payloads" "HTTP/API/Bulk files"
pipeline.orchestration.downloader -> pipeline.workspaceStorage "Writes data/{system}/acquire/{run_date} and data/{system}/prepare/{run_date}" "Filesystem (Parquet/JSON)"
pipeline.orchestration.downloader -> pipeline.orchestration.wikidataProjection "Runs the Wikidata projection during acquisition" "Python"
pipeline.orchestration.sharding -> pipeline.workspaceStorage "Reads prepared/acquired payloads and writes data/{system}/source/{run_date} shards" "Filesystem (Parquet)"
pipeline.orchestration.sharding -> pipeline.orchestration.shardingBatchProcessor "Delegates batch processing for prepared source resources" "Python"
pipeline.orchestration.shardingBatchProcessor -> pipeline.orchestration.shardingHandlers "Resolves system-specific handler dispatch and custom sharding logic" "Python"
pipeline.orchestration.sharding -> pipeline.orchestration.wikidataProjection "Runs the Wikidata projection when the source is Wikidata" "Python"
pipeline.orchestration.wikidataProjection -> pipeline.rustExtractors "Runs wikisieve as a subprocess" "Subprocess"
pipeline.orchestration.wikidataProjection -> pipeline.workspaceStorage "Reads compressed Wikidata dumps and writes projection intermediates and wikidata-run-manifest.json" "Filesystem"
pipeline.orchestration.canonicalization -> pipeline.workspaceStorage "Reads source shards and writes canonical parquet" "Filesystem (Parquet)"
pipeline.orchestration.cleansing -> pipeline.workspaceStorage "Reads canonical snapshots, writes cleansed parquet, and logs runs to mlflow.db" "Filesystem (Parquet/SQLite)"
pipeline.orchestration.cleansing -> pipeline.packages.cleanseRules "Invokes company_cleanse's cleanse_lazyframe() to normalize names and resolve company type" "Python"
pipeline.orchestration.match -> pipeline.workspaceStorage "Reads source and target canonical snapshots and writes data/{system}/matched/current with _match_metadata.json" "Filesystem (Parquet/JSON)"
pipeline.orchestration.tokenization -> pipeline.workspaceStorage "Reads cleansed parquet and writes tokenizer training corpora" "Filesystem (Parquet)"
pipeline.orchestration.tokenization -> pipeline.packages.tokenizePackage "Normalizes training names via name_preprocessing()" "Python"
pipeline.packages.tokenizePackage -> pipeline.packages.cleanseRules "Uses company_cleanse's noise-word defaults/profiles and strip_company_suffix()" "Python"

pipeline.orchestration.canonicalization -> pipeline.orchestration.registry "Uses system catalog/schema metadata" "Python"
pipeline.orchestration.downloader -> pipeline.orchestration.registry "Looks up each system's acquisition plan" "Python"
pipeline.orchestration.sharding -> pipeline.orchestration.registry "Looks up system plans and the country registry" "Python"
pipeline.analysis.analysisLibrary -> pipeline.orchestration.registry "Uses system plans to discover runnable systems" "Python"
pipeline.analysis.analysisLibrary -> pipeline.orchestration.tokenization "Reads the tokenized output column contract" "Python"

pipeline.blocking.blockingWorkflow -> pipeline.blocking.blockingContracts "Uses BlockingStrategyConfig/BlockingRunConfig" "Python"
pipeline.blocking.blockingWorkflow -> pipeline.blocking.blockingLoader "Loads both datasets per country" "Python"
pipeline.blocking.blockingWorkflow -> pipeline.blocking.blockingNameTransform "Applies the run's name transform to both sides" "Python"
pipeline.blocking.blockingWorkflow -> pipeline.blocking.blockingTruth "Reads each source row's ground truth" "Python"
pipeline.blocking.blockingWorkflow -> pipeline.blocking.blockingReporting "Builds country blocks and the recall-curve report" "Python"
pipeline.blocking.blockingWorkflow -> pipeline.blocking.blockingRunLayout "Locates the run's artefacts" "Python"
pipeline.blocking.blockingWorkflow -> pipeline.packages.vectorizePackage "Resolves target-index build settings, generates and prunes candidates via ClusteringStrategy" "Python"
pipeline.blocking.blockingWorkflow -> pipeline.packages.tokenizePackage "Tokenizes text views for token-list representations" "Python"
pipeline.blocking.blockingWorkflow -> pipeline.packages.cleanseRules "Reads effective noise words via get_effective_noise_words()" "Python"
pipeline.blocking.blockingWorkflow -> pipeline.validation.validationLibrary "Caches target indexes and scores candidates via compute_pair_truth_eval()" "Python"
pipeline.blocking.blockingWorkflow -> pipeline.workspaceLib "Caches name forms and resolves the promoted tokenizer" "Python"
pipeline.blocking.blockingNameTransform -> pipeline.validation.validationLibrary "Derives name forms via derive_name_forms()" "Python"
pipeline.blocking.blockingContracts -> pipeline.packages.vectorizePackage "Validates the representation via resolve_clustering_strategy()" "Python"
pipeline.blocking.blockingContracts -> pipeline.packages.tokenizePackage "Validates name-preprocessing profiles" "Python"
pipeline.blocking.blockingLoader -> pipeline.blocking.blockingTruth "Checks which layers carry ground truth" "Python"
pipeline.blocking.blockingLoader -> pipeline.validation.validationLibrary "Reads country partitions via read_country_partition_frame()" "Python"
pipeline.blocking.blockingLoader -> pipeline.workspaceStorage "Reads matched source and latest canonical target parquet, or a perturbed dataset's own layer" "Filesystem (Parquet)"
pipeline.blocking.blockingTruth -> pipeline.workspaceLib "Resolves cross-system matches from derived URIs" "Python"
pipeline.blocking.blockingRunLayout -> pipeline.workspaceLib "Builds blocking:// references under artifacts/blocking/" "Python"
pipeline.blocking.blockingReporting -> pipeline.blocking.blockingRunLayout "Writes each artefact where the run layout names it" "Python"
pipeline.blocking.blockingReporting -> pipeline.workspaceStorage "Writes blocking run parquet/JSON artefacts" "Filesystem (Parquet/JSON)"
pipeline.blocking.blockingStoredRuns -> pipeline.blocking.blockingRunLayout "Finds finished runs" "Python"
pipeline.blocking.blockingStoredRuns -> pipeline.blocking.blockingComparison "Builds StrategyRunEntry values from stored runs" "Python"
pipeline.blocking.blockingStoredRuns -> pipeline.workspaceStorage "Reads finished runs' records and artefacts" "Filesystem (Parquet/JSON)"
pipeline.blocking.blockingAudit -> pipeline.blocking.blockingStoredRuns "Loads the run it audits" "Python"
pipeline.blocking.blockingAudit -> pipeline.blocking.blockingWorkflow "Rescores missed pairs via rescore_missed_pairs()" "Python"
pipeline.blocking.blockingComparison -> pipeline.blocking.blockingContracts "Reads BlockingRunConfig/BlockingRunResult and the strategy_comparison schema" "Python"
pipeline.blocking.blockingInspection -> pipeline.blocking.blockingContracts "Reads a BlockingRunResult's artefact frames" "Python"
pipeline.cli -> pipeline.blocking.blockingComparison "Chains several runs into one comparison (scripts/compare_blocking_strategies.py)" "Python"
pipeline.cli -> pipeline.blocking.blockingReporting "Persists each run's artefacts and the strategy-comparison report" "Python"
pipeline.cli -> pipeline.blocking.blockingStoredRuns "Reloads stored runs for strategy-comparison reports" "Python"
pipeline.cli -> pipeline.blocking.blockingRunLayout "Locates runs and comparisons for reporting scripts" "Python"
pipeline.cli -> pipeline.blocking.blockingAudit "Reads pair audits for strategy-comparison reports" "Python"

pipeline.orchestration.downloader -> pipeline.orchestration.sharding "Provides acquired source payloads" "Filesystem (Parquet)"
pipeline.orchestration.sharding -> pipeline.orchestration.canonicalization "Provides sharded parquet outputs" "Filesystem (Parquet)"
pipeline.orchestration.canonicalization -> pipeline.orchestration.match "Provides canonical snapshots to label" "Filesystem (Parquet)"
pipeline.orchestration.cleansing -> pipeline.orchestration.tokenization "Provides cleansed names for training corpora" "Filesystem (Parquet)"
