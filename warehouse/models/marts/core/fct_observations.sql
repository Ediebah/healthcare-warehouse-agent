-- fct_observations — grain: one row per measurement (labs/vitals/survey). FKs: patient, encounter.
-- Surrogate PK uses a within-grain row_number() to stay unique when a patient has the same
-- observation code + value recorded at the same timestamp. Measure: value_numeric.

with observations as (
    select * from {{ ref('stg_observations') }}
),

keyed as (
    select
        *,
        row_number() over (
            partition by patient_id, observation_code, observed_at, value_text
            order by encounter_id, units
        ) as row_in_grain
    from observations
)

select
    {{ dbt_utils.generate_surrogate_key(['patient_id', 'observation_code', 'observed_at', 'value_text', 'row_in_grain']) }}
                                                as observation_id,
    patient_id,
    encounter_id,
    observed_at,
    observation_category,
    observation_code,
    observation_description,
    value_text,
    value_numeric,
    units,
    value_type
from keyed
