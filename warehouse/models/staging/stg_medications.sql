-- Staging: one row per medication order (RxNorm-coded). No natural PK.
-- START/STOP are full timestamps; costs and dispense count are cast to numerics.

with source as (
    select * from {{ source('synthea', 'medications') }}
),

renamed as (
    select
        PATIENT                                     as patient_id,
        nullif(ENCOUNTER, '')                       as encounter_id,
        nullif(PAYER, '')                           as payer_id,
        CODE                                        as medication_code,
        DESCRIPTION                                 as medication_description,

        cast(START as timestamp)                    as dispense_start,
        cast(nullif(STOP, '') as timestamp)         as dispense_stop,
        (nullif(STOP, '') is null)                  as is_active,

        cast(nullif(BASE_COST, '') as double)       as base_cost,
        cast(nullif(PAYER_COVERAGE, '') as double)  as payer_coverage,
        cast(nullif(DISPENSES, '') as integer)      as dispenses,
        cast(nullif(TOTALCOST, '') as double)       as total_cost,

        nullif(REASONCODE, '')                      as reason_code,
        nullif(REASONDESCRIPTION, '')               as reason_description
    from source
)

select * from renamed
