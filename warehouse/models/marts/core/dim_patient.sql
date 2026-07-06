-- dim_patient — conformed patient dimension. Grain: one row per patient. PK: patient_id.
-- Business logic that staging deliberately skipped: derived age + clinical age bands.
-- Age is measured to an "as-of" SNAPSHOT date = the latest date in the cohort (the moment the data
-- was generated), so a living patient is never older than the snapshot and age can NEVER be negative
-- -- Synthea, run in the generation year, produces newborns dated after any fixed calendar constant.
-- It stays reproducible across rebuilds (derived from the fixed-seed data, not wall-clock now).
-- Override with a fixed date via: dbt build --vars '{as_of_date: YYYY-MM-DD}'.

with patients as (
    select * from {{ ref('stg_patients') }}
),

as_of as (
    -- an explicit var override wins; otherwise the latest date present in the cohort (max of birth /
    -- death), which is >= every birth_date, so ages are guaranteed non-negative. greatest() ignores nulls.
    select coalesce(
        try_cast(nullif('{{ var("as_of_date", "") }}', '') as date),
        greatest(max(birth_date), max(death_date))
    ) as snapshot_date
    from patients
),

with_age as (
    select
        p.*,
        floor(date_diff('day', p.birth_date, coalesce(p.death_date, a.snapshot_date)) / 365.25) as age
    from patients p
    cross join as_of a
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
