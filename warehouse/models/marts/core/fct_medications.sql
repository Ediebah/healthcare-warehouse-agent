-- fct_medications — grain: one row per medication order. FKs: patient, medication, payer, encounter.
-- Surrogate PK: (patient, medication, dispense_start) is NOT unique on its own (a patient can have
-- two orders of the same drug at the same start), so we add a deterministic row_number() within the
-- grain group. This guarantees a unique PK without dropping any rows.

with medications as (
    select * from {{ ref('stg_medications') }}
),

keyed as (
    select
        *,
        row_number() over (
            partition by patient_id, medication_code, dispense_start
            order by encounter_id, dispense_stop, total_cost, dispenses
        ) as row_in_grain
    from medications
)

select
    {{ dbt_utils.generate_surrogate_key(['patient_id', 'medication_code', 'dispense_start', 'row_in_grain']) }}
                                                        as medication_order_id,
    patient_id,
    medication_code,
    payer_id,
    encounter_id,
    dispense_start,
    dispense_stop,
    is_active,
    base_cost,
    payer_coverage,
    dispenses,
    total_cost,
    date_diff('day', dispense_start, dispense_stop)     as days_supplied   -- null if ongoing
from keyed
