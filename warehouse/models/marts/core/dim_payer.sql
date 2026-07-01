-- dim_payer — insurance payers. Grain: one row per payer_id. PK: payer_id.

with payers as (
    select * from {{ ref('stg_payers') }}
)

select
    payer_id,
    payer_name,
    ownership,
    state_headquartered,
    amount_covered,
    amount_uncovered,
    revenue,
    covered_encounters,
    uncovered_encounters
from payers
