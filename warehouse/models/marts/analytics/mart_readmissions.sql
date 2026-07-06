-- mart_readmissions — 30-day inpatient readmission flags.
-- Grain: one row per INDEX inpatient encounter. PK: index_encounter_id.
-- Definition: an index inpatient stay is "readmitted" if the SAME patient has a subsequent inpatient
-- admission that starts STRICTLY AFTER this stay's discharge AND within a 1-30 day window. Uses lead()
-- over admission time to find the next admission per patient.
-- Why the strictly-after guard: encounter_start/stop are TIMESTAMPs and date_diff('day', ...) counts
-- calendar-day boundaries crossed, NOT signed elapsed time. A plain `between 0 and 30` therefore
-- mis-counts same-day transfers and overlapping/nested stays whose next admission actually starts
-- BEFORE discharge (elapsed time negative, but date_diff('day', ...) reads 0). Comparing the raw
-- timestamps (next_admission_start > encounter_stop) excludes those sub-day negatives.

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
        -- Next admission in time order. Tiebreak on stop then id so overlapping/nested stays that
        -- share a start ordinate resolve deterministically (shorter/earlier-ending stay first).
        lead(encounter_start) over (
            partition by patient_id
            order by encounter_start, encounter_stop, encounter_id
        ) as next_admission_start
    from inpatient
)

select
    encounter_id                                                as index_encounter_id,
    patient_id,
    encounter_start                                             as admission_date,
    encounter_stop                                              as discharge_date,
    next_admission_start,
    -- only a genuine post-discharge gap; overlapping / concurrent encounters (next admission starts
    -- before discharge) have no meaningful "days to next", so leave null rather than expose a negative.
    case when next_admission_start > encounter_stop
         then date_diff('day', encounter_stop, next_admission_start)
    end                                                         as days_to_next_admission,
    -- Readmission = next inpatient admission starts strictly AFTER discharge (timestamp compare, so
    -- sub-day negatives / same-day transfers are excluded) AND lands within a 1-30 day window.
    coalesce(
        next_admission_start > encounter_stop
        and date_diff('day', encounter_stop, next_admission_start) between 1 and 30,
        false
    )                                                           as is_30d_readmission,
    total_claim_cost
from sequenced
