// Viewset-scoped: excludes the last-modified date/version metadata from every rendered
// diagram. Distinct from the element-style "metadata" boolean, which controls the
// [Container]/[Component] type labels inside element boxes.
properties {
    "structurizr.metadata" "false"
}

systemContext pipeline "SystemContext" {
    include dataEngineer
    include dataAnalyst
    include upstreamRegistries
    include downstreamConsumers
    include externalConsumers
    include pipeline
    autolayout lr
}

container pipeline "Containers" {
    include dataEngineer
    include dataAnalyst
    include upstreamRegistries
    include downstreamConsumers
    include externalConsumers
    include pipeline.cli
    include pipeline.orchestration
    include pipeline.analysis
    include pipeline.training
    include pipeline.blocking
    include pipeline.validation
    include pipeline.packages
    include pipeline.rustExtractors
    include pipeline.workspaceLib
    include pipeline.workspaceStorage
    include pipeline.notebooks
    autolayout lr
}

container pipeline "AnalysisContainers" {
    include dataAnalyst
    include downstreamConsumers
    include pipeline.analysis
    include pipeline.workspaceStorage
    autolayout lr
}

container pipeline "TrainingContainers" {
    include dataEngineer
    include pipeline.cli
    include pipeline.training
    include pipeline.packages
    include pipeline.workspaceStorage
    autolayout lr
}

container pipeline "ValidationContainers" {
    include dataEngineer
    include pipeline.cli
    include pipeline.blocking
    include pipeline.validation
    include pipeline.packages
    include pipeline.workspaceStorage
    autolayout lr
}

component pipeline.analysis "AnalysisComponents" {
    include dataAnalyst
    include downstreamConsumers
    include pipeline.analysis.analysisCli
    include pipeline.analysis.analysisLibrary
    include pipeline.notebooks
    include pipeline.workspaceStorage
    autolayout tb
}

component pipeline.training "TrainingComponents" {
    include dataEngineer
    include pipeline.cli
    include pipeline.training.trainCli
    include pipeline.training.trainLibrary
    include pipeline.packages.tokenizePackage
    include pipeline.packages.vectorizePackage
    include pipeline.workspaceStorage
    autolayout tb
}

component pipeline.validation "ValidationComponents" {
    include dataEngineer
    include pipeline.cli
    include pipeline.validation.validationCli
    include pipeline.validation.validationLibrary
    include pipeline.packages.cleanseRules
    include pipeline.packages.tokenizePackage
    include pipeline.packages.vectorizePackage
    include pipeline.packages.classifyPackage
    include pipeline.packages.perturbationPackage
    include pipeline.workspaceStorage
    autolayout tb
}

component pipeline.blocking "BlockingComponents" {
    include dataEngineer
    include pipeline.blocking.blockingContracts
    include pipeline.blocking.blockingLoader
    include pipeline.blocking.blockingNameTransform
    include pipeline.blocking.blockingTruth
    include pipeline.blocking.blockingWorkflow
    include pipeline.blocking.blockingRunLayout
    include pipeline.blocking.blockingReporting
    include pipeline.blocking.blockingStoredRuns
    include pipeline.blocking.blockingComparison
    include pipeline.blocking.blockingAudit
    include pipeline.blocking.blockingInspection
    include pipeline.cli
    include pipeline.packages.cleanseRules
    include pipeline.packages.tokenizePackage
    include pipeline.packages.vectorizePackage
    include pipeline.validation
    include pipeline.workspaceLib
    include pipeline.workspaceStorage
    autolayout tb
}

component pipeline.orchestration "OrchestrationComponents" {
    include dataEngineer
    include pipeline.cli
    include upstreamRegistries
    include pipeline.workspaceStorage
    include pipeline.packages.cleanseRules
    include pipeline.packages.tokenizePackage
    include pipeline.orchestration.downloader
    include pipeline.orchestration.sharding
    include pipeline.orchestration.shardingBatchProcessor
    include pipeline.orchestration.shardingHandlers
    include pipeline.orchestration.wikidataProjection
    include pipeline.rustExtractors
    include pipeline.orchestration.canonicalization
    include pipeline.orchestration.cleansing
    include pipeline.orchestration.match
    include pipeline.orchestration.tokenization
    include pipeline.orchestration.registry
    autolayout tb
}

component pipeline.orchestration "ShardingComponents" {
    include dataEngineer
    include pipeline.cli
    include upstreamRegistries
    include pipeline.workspaceStorage
    include pipeline.orchestration.downloader
    include pipeline.orchestration.sharding
    include pipeline.orchestration.shardingBatchProcessor
    include pipeline.orchestration.shardingHandlers
    include pipeline.orchestration.wikidataProjection
    include pipeline.rustExtractors
    include pipeline.orchestration.registry
    autolayout tb
}

component pipeline.packages "PackagesComponents" {
    include externalConsumers
    include downstreamConsumers
    include pipeline.cli
    include pipeline.orchestration
    include pipeline.analysis
    include pipeline.notebooks
    include pipeline.training
    include pipeline.validation
    include pipeline.blocking
    include pipeline.packages.cleanseRules
    include pipeline.packages.tokenizePackage
    include pipeline.packages.vectorizePackage
    include pipeline.packages.classifyPackage
    include pipeline.packages.perturbationPackage
    include pipeline.packages.resolversPackage
    autolayout tb
}

// Palette per docs/diagrams/style-guide.md. ACCENT and MUTED come from _palette.dsl, which
// tooling/sync_palette.py generates from docs/diagrams/house-style.css; the rest of the
// C4 element ramp is Structurizr's own.
styles {
    element "Person" {
        shape person
        background #0B3C73
        color #ffffff
    }
    element "Software System" {
        background #1A4F8B
        color #ffffff
    }
    // defined after "Software System" so it wins for the external ones: C4 convention is
    // that anything outside the system in scope reads as muted/grey
    element "External" {
        background ${MUTED}
        color #ffffff
    }
    element "Container" {
        background ${ACCENT}
        color #ffffff
    }
    element "Component" {
        background #7FA3CC
        color #111111
    }
    relationship "Planned" {
        style dashed
        color #888888
    }
}

theme default
