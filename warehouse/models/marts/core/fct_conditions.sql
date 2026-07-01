-- fct_conditions — grain: one row per patient-condition episode. No natural key, so we mint a
-- deterministic surrogate PK from the grain columns. FKs: patient, condition, encounter.

with conditions as (
    select * from {{ ref('stg_conditions') }}
)

select
    {{ dbt_utils.generate_surrogate_key(['patient_id', 'condition_code', 'onset_date']) }}
                                                    as condition_episode_id,
    patient_id,
    condition_code,
    encounter_id,
    onset_date,
    resolved_date,
    is_active,
    date_diff('day', onset_date, resolved_date)     as duration_days   -- null if unresolved
from conditions
