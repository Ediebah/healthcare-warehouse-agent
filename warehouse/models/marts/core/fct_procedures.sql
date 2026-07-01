-- fct_procedures — grain: one row per procedure performed. FKs: patient, procedure, encounter.
-- Surrogate PK uses a within-grain row_number() to stay unique when a patient has the same
-- procedure code recorded twice at the same start time. Measure: base_cost.

with procedures as (
    select * from {{ ref('stg_procedures') }}
),

keyed as (
    select
        *,
        row_number() over (
            partition by patient_id, procedure_code, procedure_start
            order by encounter_id, base_cost
        ) as row_in_grain
    from procedures
)

select
    {{ dbt_utils.generate_surrogate_key(['patient_id', 'procedure_code', 'procedure_start', 'row_in_grain']) }}
                                                    as procedure_event_id,
    patient_id,
    procedure_code,
    encounter_id,
    procedure_start,
    procedure_stop,
    base_cost,
    reason_code,
    reason_description
from keyed
