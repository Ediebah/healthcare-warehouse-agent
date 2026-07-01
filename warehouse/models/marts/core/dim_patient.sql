-- dim_patient — conformed patient dimension. Grain: one row per patient. PK: patient_id.
-- Business logic that staging deliberately skipped: derived age + clinical age bands.
-- Age uses coalesce(death_date, current_date), so living patients' age refreshes over time
-- (a determinism trade-off we accept; swap current_date for a fixed date to freeze it).

with patients as (
    select * from {{ ref('stg_patients') }}
),

with_age as (
    select
        *,
        floor(date_diff('day', birth_date, coalesce(death_date, current_date)) / 365.25) as age
    from patients
)

select
    patient_id,
    birth_date,
    death_date,
    is_deceased,
    age,
    case
        when age < 18 then '0-17'
        when age < 40 then '18-39'
        when age < 65 then '40-64'
        when age < 75 then '65-74'
        else '75+'
    end                                     as age_group,
    gender,
    race,
    ethnicity,
    marital_status,
    city,
    state,
    county,
    zip_code,
    healthcare_expenses,
    healthcare_coverage,
    income
from with_age
