-- Staging: labs/vitals/survey results in tall format — one row per measurement (LOINC-coded).
-- VALUE is mixed numeric/text; we keep the raw text and TRY_CAST a numeric copy (null if non-numeric).

with source as (
    select * from {{ source('synthea', 'observations') }}
),

renamed as (
    select
        PATIENT                             as patient_id,
        nullif(ENCOUNTER, '')               as encounter_id,
        cast(DATE as timestamp)             as observed_at,
        CATEGORY                            as observation_category,
        CODE                                as observation_code,
        DESCRIPTION                         as observation_description,
        VALUE                               as value_text,
        try_cast(VALUE as double)           as value_numeric,
        nullif(UNITS, '')                   as units,
        TYPE                                as value_type
    from source
)

select * from renamed
