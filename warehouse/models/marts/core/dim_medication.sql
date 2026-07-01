-- dim_medication — distinct medication concepts. Grain: one row per medication_code (RxNorm).
-- PK: medication_code.

with medications as (
    select * from {{ ref('stg_medications') }}
)

select
    medication_code,
    max(medication_description) as medication_description
from medications
group by medication_code
