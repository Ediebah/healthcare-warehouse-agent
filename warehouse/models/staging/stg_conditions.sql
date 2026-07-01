-- Staging: one row per patient-condition episode (SNOMED-coded).
-- START = onset date, STOP = resolution date (null if unresolved/chronic). No natural PK.

with source as (
    select * from {{ source('synthea', 'conditions') }}
),

renamed as (
    select
        PATIENT                             as patient_id,
        nullif(ENCOUNTER, '')               as encounter_id,
        SYSTEM                              as code_system,
        CODE                                as condition_code,
        DESCRIPTION                         as condition_description,
        cast(START as date)                 as onset_date,
        cast(nullif(STOP, '') as date)      as resolved_date,
        (nullif(STOP, '') is null)          as is_active
    from source
)

select * from renamed
