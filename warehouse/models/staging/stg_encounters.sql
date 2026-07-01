-- Staging: one row per encounter (visit). Grain: encounter_id.
-- Casts timestamps and cost columns; keeps FKs to patient/org/provider/payer.

with source as (
    select * from {{ source('synthea', 'encounters') }}
),

renamed as (
    select
        Id                                                  as encounter_id,

        -- foreign keys
        PATIENT                                             as patient_id,
        nullif(ORGANIZATION, '')                            as organization_id,
        nullif(PROVIDER, '')                                as provider_id,
        nullif(PAYER, '')                                   as payer_id,

        -- timing
        cast(START as timestamp)                            as encounter_start,
        cast(nullif(STOP, '') as timestamp)                 as encounter_stop,
        cast(START as date)                                 as encounter_date,

        -- classification
        ENCOUNTERCLASS                                      as encounter_class,
        CODE                                                as encounter_code,
        DESCRIPTION                                         as encounter_description,
        nullif(REASONCODE, '')                              as reason_code,
        nullif(REASONDESCRIPTION, '')                       as reason_description,

        -- costs
        cast(nullif(BASE_ENCOUNTER_COST, '') as double)     as base_encounter_cost,
        cast(nullif(TOTAL_CLAIM_COST, '') as double)        as total_claim_cost,
        cast(nullif(PAYER_COVERAGE, '') as double)          as payer_coverage
    from source
)

select * from renamed
