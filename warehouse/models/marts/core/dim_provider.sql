-- dim_provider — clinicians. Grain: one row per provider_id. PK: provider_id.
-- Demonstrates a marts-layer JOIN: denormalize the organization NAME onto the provider row
-- so downstream queries don't need a second hop. (Staging never joins; marts do.)

with providers as (
    select * from {{ ref('stg_providers') }}
),

organizations as (
    select organization_id, organization_name
    from {{ ref('stg_organizations') }}
)

select
    p.provider_id,
    p.provider_name,
    p.gender,
    p.specialty,
    p.organization_id,
    o.organization_name,
    p.city,
    p.state,
    p.zip_code,
    p.encounter_count,
    p.procedure_count
from providers p
left join organizations o using (organization_id)
