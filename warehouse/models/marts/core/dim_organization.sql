-- dim_organization — healthcare organizations. Grain: one row per organization_id.
-- Essentially a pass-through of staging, promoted to a materialized table in the star schema.

with organizations as (
    select * from {{ ref('stg_organizations') }}
)

select
    organization_id,
    organization_name,
    city,
    state,
    zip_code,
    latitude,
    longitude,
    revenue,
    utilization
from organizations
