-- mart_readmissions — 30-day inpatient readmission flags.
-- Grain: one row per INDEX inpatient encounter. PK: index_encounter_id.
-- Definition: an index inpatient stay is "readmitted" if the SAME patient has another inpatient
-- admission whose start is 0-30 days after this stay's discharge. Uses lead() to find the next
-- admission per patient in time order.

with inpatient as (
    select
        encounter_id,
        patient_id,
        encounter_start,
        encounter_stop,
        total_claim_cost
    from {{ ref('fct_encounters') }}
    where encounter_class = 'inpatient'
),

sequenced as (
    select
        *,
        lead(encounter_start) over (
            partition by patient_id
            order by encounter_start
        ) as next_admission_start
    from inpatient
)

select
    encounter_id                                                as index_encounter_id,
    patient_id,
    encounter_start                                             as admission_date,
    encounter_stop                                              as discharge_date,
    next_admission_start,
    date_diff('day', encounter_stop, next_admission_start)      as days_to_next_admission,
    coalesce(
        date_diff('day', encounter_stop, next_admission_start) between 0 and 30,
        false
    )                                                           as is_30d_readmission,
    total_claim_cost
from sequenced
