workspace "CompanyBlocker Architecture" "An exploratory research platform for blocking algorithms on company names, spanning acquisition, normalization, cleansing, tokenization, vectorization/blocking, training, and validation across multi-system company registries." {
    # The prose architecture docs and decision records are attached to the CompanyBlocker
    # software system in model.dsl, not here: the model.softwaresystem.documentation and
    # .decisions inspections check that element, and a workspace-level !docs fills the
    # tabs while still failing them.
    model {
        !include model.dsl
    }

    views {
        !include _palette.dsl
        !include views.dsl
    }

    configuration {
        scope softwaresystem
    }
}