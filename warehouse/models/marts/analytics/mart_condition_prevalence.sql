-- mart_condition_prevalence — condition prevalence by patient age band.
-- Grain: one row per (condition_code, age_group). Composite key tested for uniqueness.
--
-- prevalence_pct = 100 * (distinct patients in the age band who have the condition)
--                        / (all patients in that age band).
-- CAVEAT (baked in for the guardrail): denominators for small bands (e.g. 75+, n=91) make rare-
-- condition prevalence estimates noisy. `total_patients_in_age_group` is exposed so a consumer
-- can judge reliability / compute a confidence interval.

with patient_dim as (
    select patient_id, age_group
    from {{ ref('dim_patient') }}
),

-- distinct patient↔condition pairs (a patient with a condition counts once)
patient_conditions as (
    select distinct patient_id, condition_code
    from {{ ref('fct_conditions') }}
),

-- denominator: how many patients are in each age band
group_totals as (
    select age_group, count(*) as total_patients
    from patient_dim
    group by age_group
),

-- numerator: distinct patients with each condition, per age band
numerator as (
    select
        pc.condition_code,
        pd.age_group,
        count(distinct pc.patient_id) as patients_with_condition
    from patient_conditions pc
    join patient_dim pd using (patient_id)
    group by pc.condition_code, pd.age_group
)

select
    n.condition_code,
    dc.condition_description,
    n.age_group,
    n.patients_with_condition,
    g.total_patients                                                   as total_patients_in_age_group,
    round(100.0 * n.patients_with_condition / g.total_patients, 2)     as prevalence_pct
from numerator n
join group_totals g using (age_group)
left join {{ ref('dim_condition') }} dc using (condition_code)
