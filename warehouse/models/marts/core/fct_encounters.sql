-- fct_encounters — grain: one row per encounter (visit). PK: encounter_id (natural).
-- Foreign keys to patient/organization/provider/payer dims. Measures: costs + duration.

with encounters as (
    select * from {{ ref('stg_encounters') }}
)

select
    -- keys
    encounter_id,
    patient_id,
    organization_id,
    provider_id,
    payer_id,

    -- attributes
    encounter_date,
    encounter_start,
    encounter_stop,
    encounter_class,
    encounter_code,
    encounter_description,
    reason_code,
    reason_description,

    -- measures
    base_encounter_cost,
    total_claim_cost,
    payer_coverage,
    total_claim_cost - payer_coverage                       as patient_out_of_pocket,
    date_diff('minute', encounter_start, encounter_stop)    as duration_minutes
from encounters
