-- Staging: one row per patient. Rename to house style, cast types, derive a deceased flag.
-- Drops raw PII identifiers (SSN/license/passport) that carry no analytic value.

with source as (
    select * from {{ source('synthea', 'patients') }}
),

renamed as (
    select
        Id                                                as patient_id,

        -- dates
        cast(BIRTHDATE as date)                           as birth_date,
        cast(nullif(DEATHDATE, '') as date)               as death_date,
        (nullif(DEATHDATE, '') is not null)               as is_deceased,

        -- demographics
        GENDER                                            as gender,
        RACE                                              as race,
        ETHNICITY                                         as ethnicity,
        nullif(MARITAL, '')                               as marital_status,
        FIRST                                             as first_name,
        LAST                                              as last_name,

        -- geography
        BIRTHPLACE                                        as birthplace,
        CITY                                              as city,
        STATE                                             as state,
        COUNTY                                            as county,
        ZIP                                               as zip_code,
        cast(nullif(LAT, '') as double)                   as latitude,
        cast(nullif(LON, '') as double)                   as longitude,

        -- lifetime cost/income totals (synthetic)
        cast(nullif(HEALTHCARE_EXPENSES, '') as double)   as healthcare_expenses,
        cast(nullif(HEALTHCARE_COVERAGE, '') as double)   as healthcare_coverage,
        cast(nullif(INCOME, '') as double)                as income
    from source
)

select * from renamed
