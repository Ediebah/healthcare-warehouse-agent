-- Staging: one row per insurance payer.

with source as (
    select * from {{ source('synthea', 'payers') }}
),

renamed as (
    select
        Id                                              as payer_id,
        NAME                                            as payer_name,
        nullif(OWNERSHIP, '')                           as ownership,
        STATE_HEADQUARTERED                             as state_headquartered,
        cast(nullif(AMOUNT_COVERED, '') as double)      as amount_covered,
        cast(nullif(AMOUNT_UNCOVERED, '') as double)    as amount_uncovered,
        cast(nullif(REVENUE, '') as double)             as revenue,
        cast(nullif(COVERED_ENCOUNTERS, '') as integer)   as covered_encounters,
        cast(nullif(UNCOVERED_ENCOUNTERS, '') as integer) as uncovered_encounters
    from source
)

select * from renamed
