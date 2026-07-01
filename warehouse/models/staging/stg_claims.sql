-- Staging: one row per insurance claim header. We take a thin, useful slice of the 31 raw
-- columns (claim identity, patient/provider/insurer FKs, primary diagnosis, service date).

with source as (
    select * from {{ source('synthea', 'claims') }}
),

renamed as (
    select
        Id                                              as claim_id,
        PATIENTID                                       as patient_id,
        nullif(PROVIDERID, '')                          as provider_id,
        nullif(PRIMARYPATIENTINSURANCEID, '')           as primary_payer_id,
        nullif(SECONDARYPATIENTINSURANCEID, '')         as secondary_payer_id,
        nullif(DIAGNOSIS1, '')                          as primary_diagnosis_code,
        cast(nullif(SERVICEDATE, '') as timestamp)      as service_date,
        cast(nullif(CURRENTILLNESSDATE, '') as timestamp) as current_illness_date,
        nullif(STATUSP, '')                             as primary_status
    from source
)

select * from renamed
