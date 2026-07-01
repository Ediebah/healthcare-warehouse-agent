-- mart_cost_by_condition — cost associated with each condition.
-- Grain: one row per condition_code. PK: condition_code.
--
-- HONEST DEFINITION: this is the cost of the ENCOUNTER AT WHICH the condition was recorded,
-- aggregated by condition — NOT the total lifetime cost of treating the condition (that would
-- require attributing downstream encounters/meds, and invites confounding). Column names say so.
-- The Weekend-2 guardrail step should flag any causal reading of these numbers.

with condition_episodes as (
    select condition_code, patient_id, encounter_id
    from {{ ref('fct_conditions') }}
),

encounter_cost as (
    select encounter_id, total_claim_cost
    from {{ ref('fct_encounters') }}
),

joined as (
    select
        ce.condition_code,
        ce.patient_id,
        ec.total_claim_cost
    from condition_episodes ce
    left join encounter_cost ec using (encounter_id)
)

select
    j.condition_code,
    dc.condition_description,
    count(*)                                    as num_episodes,
    count(distinct j.patient_id)                as num_patients,
    round(sum(j.total_claim_cost), 2)           as total_diagnosing_encounter_cost,
    round(avg(j.total_claim_cost), 2)           as avg_diagnosing_encounter_cost
from joined j
left join {{ ref('dim_condition') }} dc using (condition_code)
group by 1, 2
