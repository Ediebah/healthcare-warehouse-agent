-- Staging: one row per clinician, linked to an organization.
-- Note: raw column SPECIALITY (Synthea's spelling) is normalized to `specialty`.

with source as (
    select * from {{ source('synthea', 'providers') }}
),

renamed as (
    select
        Id                                      as provider_id,
        ORGANIZATION                            as organization_id,
        NAME                                    as provider_name,
        GENDER                                  as gender,
        SPECIALITY                              as specialty,
        CITY                                    as city,
        STATE                                   as state,
        ZIP                                     as zip_code,
        cast(nullif(ENCOUNTERS, '') as integer) as encounter_count,
        cast(nullif(PROCEDURES, '') as integer) as procedure_count
    from source
)

select * from renamed
