-- dim_condition — distinct condition concepts. Grain: one row per condition_code (SNOMED-CT).
-- PK: condition_code. A code appears many times across patients in the facts; here it's unique.

with conditions as (
    select * from {{ ref('stg_conditions') }}
)

select
    condition_code,
    max(condition_description) as condition_description,   -- constant per code; max() picks it
    max(code_system)           as code_system
from conditions
group by condition_code
