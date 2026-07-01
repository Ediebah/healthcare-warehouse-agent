-- Staging: one row per healthcare organization (hospital/clinic).

with source as (
    select * from {{ source('synthea', 'organizations') }}
),

renamed as (
    select
        Id                                          as organization_id,
        NAME                                        as organization_name,
        CITY                                        as city,
        STATE                                       as state,
        ZIP                                         as zip_code,
        cast(nullif(LAT, '') as double)             as latitude,
        cast(nullif(LON, '') as double)             as longitude,
        nullif(PHONE, '')                           as phone,
        cast(nullif(REVENUE, '') as double)         as revenue,
        cast(nullif(UTILIZATION, '') as integer)    as utilization
    from source
)

select * from renamed
