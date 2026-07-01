-- Staging: one row per procedure performed (SNOMED-coded). No natural PK.

with source as (
    select * from {{ source('synthea', 'procedures') }}
),

renamed as (
    select
        PATIENT                                 as patient_id,
        nullif(ENCOUNTER, '')                   as encounter_id,
        SYSTEM                                  as code_system,
        CODE                                    as procedure_code,
        DESCRIPTION                             as procedure_description,
        cast(START as timestamp)                as procedure_start,
        cast(nullif(STOP, '') as timestamp)     as procedure_stop,
        cast(nullif(BASE_COST, '') as double)   as base_cost,
        nullif(REASONCODE, '')                  as reason_code,
        nullif(REASONDESCRIPTION, '')           as reason_description
    from source
)

select * from renamed
