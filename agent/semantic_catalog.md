# Semantic Catalog

Generated from dbt artifacts. 14 tables, 6 named metrics.

## Named metrics

### `readmission_rate_30d`
- **Definition:** Share of index inpatient stays followed by another inpatient admission within 0-30 days of discharge.
- **Source model:** `mart_readmissions`
- **SQL:** `avg(is_30d_readmission::int)  -- optionally *100 for a percentage`
- **Caveats:** Inpatient-only. Denominator is index inpatient stays, not patients.

### `avg_encounter_cost`
- **Definition:** Average total billed cost across encounters.
- **Source model:** `fct_encounters`
- **SQL:** `avg(total_claim_cost)`
- **Caveats:** Billed (claim) cost, synthetic. Segment by encounter_class for fair comparisons.

### `avg_patient_out_of_pocket`
- **Definition:** Average amount the patient pays after insurance, per encounter.
- **Source model:** `fct_encounters`
- **SQL:** `avg(patient_out_of_pocket)  -- total_claim_cost - payer_coverage`
- **Caveats:** Synthetic payer logic; not real benefit design.

### `condition_prevalence_by_age`
- **Definition:** Percent of patients in an age band who have a given condition.
- **Source model:** `mart_condition_prevalence`
- **SQL:** `prevalence_pct  -- 100 * patients_with_condition / total_patients_in_age_group`
- **Caveats:** Small age bands (e.g. 75+, n≈91) give noisy estimates — check total_patients_in_age_group.

### `avg_diagnosing_encounter_cost_by_condition`
- **Definition:** Average cost of the encounter at which a condition was diagnosed, per condition.
- **Source model:** `mart_cost_by_condition`
- **SQL:** `avg_diagnosing_encounter_cost`
- **Caveats:** NOT lifetime cost of treating the condition — no causal attribution of downstream care.

### `total_medication_cost`
- **Definition:** Total cost of medication orders.
- **Source model:** `fct_medications`
- **SQL:** `sum(total_cost)`
- **Caveats:** Order-level synthetic cost; a patient may have many orders of one drug.

## Tables

### `dim_condition`  (core)
Condition-concept dimension. Grain: one row per SNOMED-CT condition_code.

- **Relation:** `healthcare.main.dim_condition`
- **Primary key:** condition_code

| column | type | description | examples |
|---|---|---|---|
| `condition_code` | VARCHAR | Primary key (SNOMED-CT code). | 160904001, 73438004, 427419006 |
| `condition_description` | VARCHAR |  | Pneumonia (disorder), Proliferative diabetic retinopathy due to type II diabetes mellitus, Mitral valve regurgitation (disorder) |
| `code_system` | VARCHAR |  | SNOMED-CT |

### `dim_medication`  (core)
Medication-concept dimension. Grain: one row per RxNorm medication_code.

- **Relation:** `healthcare.main.dim_medication`
- **Primary key:** medication_code

| column | type | description | examples |
|---|---|---|---|
| `medication_code` | VARCHAR | Primary key (RxNorm code). | 313110, 106892, 562251 |
| `medication_description` | VARCHAR |  | Meperidine Hydrochloride 50 MG Oral Tablet, fosfomycin 3000 MG Granules for Oral Solution, amoxicillin 875 MG / clavulanate 125 MG Oral Tablet |

### `dim_organization`  (core)
Organization dimension. Grain: one row per organization_id.

- **Relation:** `healthcare.main.dim_organization`
- **Primary key:** organization_id

| column | type | description | examples |
|---|---|---|---|
| `organization_id` | VARCHAR | Primary key (UUID). |  |
| `organization_name` | VARCHAR |  | TRINITY FAMILY MEDICINE, CAREWELL URGENT CARE CENTERS OF MA  PC, CLAFLIN HILL CORPORATION |
| `city` | VARCHAR |  | BRIDGEWATER, FALMOUTH, NEEDHAM |
| `state` | VARCHAR |  | MA |
| `zip_code` | VARCHAR |  | 025328305, 014535768, 018033735 |
| `latitude` | DOUBLE |  |  |
| `longitude` | DOUBLE |  |  |
| `revenue` | DOUBLE |  |  |
| `utilization` | INTEGER |  |  |

### `dim_patient`  (core)
Patient dimension. Grain: one row per patient. Adds derived age + clinical age bands.

- **Relation:** `healthcare.main.dim_patient`
- **Primary key:** patient_id

| column | type | description | examples |
|---|---|---|---|
| `patient_id` | VARCHAR | Primary key (UUID). |  |
| `birth_date` | DATE |  |  |
| `death_date` | DATE |  |  |
| `is_deceased` | BOOLEAN |  |  |
| `age` | DOUBLE | Age in years at death (if deceased) or as of the current date. |  |
| `age_group` | VARCHAR | Clinical age band. | 0-17, 18-39, 65-74 |
| `gender` | VARCHAR |  | M, F |
| `race` | VARCHAR |  | white, native, asian |
| `ethnicity` | VARCHAR |  | nonhispanic, hispanic |
| `marital_status` | VARCHAR |  | M, W, D |
| `city` | VARCHAR |  | Somerville, Hopkinton, Hudson |
| `state` | VARCHAR |  | Massachusetts |
| `county` | VARCHAR |  | Suffolk County, Worcester County, Nantucket County |
| `zip_code` | VARCHAR |  | 02134, 02155, 01904 |
| `healthcare_expenses` | DOUBLE |  |  |
| `healthcare_coverage` | DOUBLE |  |  |
| `income` | DOUBLE |  |  |

### `dim_payer`  (core)
Payer dimension. Grain: one row per payer_id.

- **Relation:** `healthcare.main.dim_payer`
- **Primary key:** payer_id

| column | type | description | examples |
|---|---|---|---|
| `payer_id` | VARCHAR | Primary key (UUID). |  |
| `payer_name` | VARCHAR |  | Medicare, Blue Cross Blue Shield, UnitedHealthcare |
| `ownership` | VARCHAR |  | PRIVATE, GOVERNMENT, NO_INSURANCE |
| `state_headquartered` | VARCHAR |  |  |
| `amount_covered` | DOUBLE |  |  |
| `amount_uncovered` | DOUBLE |  |  |
| `revenue` | DOUBLE |  |  |
| `covered_encounters` | INTEGER |  |  |
| `uncovered_encounters` | INTEGER |  |  |

### `dim_provider`  (core)
Provider dimension with denormalized organization name. Grain: one row per provider_id.

- **Relation:** `healthcare.main.dim_provider`
- **Primary key:** provider_id

| column | type | description | examples |
|---|---|---|---|
| `provider_id` | VARCHAR | Primary key (UUID). |  |
| `provider_name` | VARCHAR |  | Huong243 Jakubowski832, Jasper743 Champlin946, Carylon722 Corwin846 |
| `gender` | VARCHAR |  | M, F |
| `specialty` | VARCHAR |  | GENERAL PRACTICE |
| `organization_id` | VARCHAR |  |  |
| `organization_name` | VARCHAR |  | BOSTON HEALTH CARE FOR THE HOMELESS PROGRAM INC, NURSE ON CALL, CAPE HERITAGE REHABILITATION & HEALTH CARE CENTER |
| `city` | VARCHAR |  | NORTH READING, SHARON, BOSTON |
| `state` | VARCHAR |  | MA |
| `zip_code` | VARCHAR |  | 025328305, 014535768, 018033735 |
| `encounter_count` | INTEGER |  |  |
| `procedure_count` | INTEGER |  |  |

### `fct_conditions`  (core)
Condition-episode fact. Grain: one row per patient-condition episode.

- **Relation:** `healthcare.main.fct_conditions`
- **Primary key:** condition_episode_id
- **Foreign keys:** patient_id → dim_patient.patient_id; condition_code → dim_condition.condition_code

| column | type | description | examples |
|---|---|---|---|
| `condition_episode_id` | VARCHAR | Surrogate primary key. |  |
| `patient_id` | VARCHAR |  |  |
| `condition_code` | VARCHAR |  | 160904001, 73438004, 427419006 |
| `encounter_id` | VARCHAR |  |  |
| `onset_date` | DATE |  |  |
| `resolved_date` | DATE |  |  |
| `is_active` | BOOLEAN |  |  |
| `duration_days` | BIGINT |  |  |

### `fct_encounters`  (core)
Encounter fact. Grain: one row per encounter. Measures: costs, out-of-pocket, duration.

- **Relation:** `healthcare.main.fct_encounters`
- **Primary key:** encounter_id
- **Foreign keys:** patient_id → dim_patient.patient_id; organization_id → dim_organization.organization_id; provider_id → dim_provider.provider_id; payer_id → dim_payer.payer_id

| column | type | description | examples |
|---|---|---|---|
| `encounter_id` | VARCHAR | Primary key (natural). |  |
| `patient_id` | VARCHAR |  |  |
| `organization_id` | VARCHAR |  |  |
| `provider_id` | VARCHAR |  |  |
| `payer_id` | VARCHAR |  |  |
| `encounter_date` | DATE |  |  |
| `encounter_start` | TIMESTAMP |  |  |
| `encounter_stop` | TIMESTAMP |  |  |
| `encounter_class` | VARCHAR |  | ambulatory, virtual, wellness |
| `encounter_code` | VARCHAR |  | 305408004, 453131000124105, 185349003 |
| `encounter_description` | VARCHAR |  | Well child visit (procedure), Non-urgent orthopedic admission (procedure), Urgent care clinic (environment) |
| `reason_code` | VARCHAR |  | 128613002, 37849005, 37320007 |
| `reason_description` | VARCHAR |  | Seizure disorder (disorder), Mitral valve regurgitation (disorder), Idiopathic atrophic hypothyroidism (disorder) |
| `base_encounter_cost` | DOUBLE |  |  |
| `total_claim_cost` | DOUBLE |  |  |
| `payer_coverage` | DOUBLE |  |  |
| `patient_out_of_pocket` | DOUBLE |  |  |
| `duration_minutes` | BIGINT |  |  |

### `fct_medications`  (core)
Medication-order fact. Grain: one row per medication order.

- **Relation:** `healthcare.main.fct_medications`
- **Primary key:** medication_order_id
- **Foreign keys:** patient_id → dim_patient.patient_id; medication_code → dim_medication.medication_code

| column | type | description | examples |
|---|---|---|---|
| `medication_order_id` | VARCHAR | Surrogate primary key. |  |
| `patient_id` | VARCHAR |  |  |
| `medication_code` | VARCHAR |  | 106892, 562251, 106258 |
| `payer_id` | VARCHAR |  |  |
| `encounter_id` | VARCHAR |  |  |
| `dispense_start` | TIMESTAMP |  |  |
| `dispense_stop` | TIMESTAMP |  |  |
| `is_active` | BOOLEAN |  |  |
| `base_cost` | DOUBLE |  |  |
| `payer_coverage` | DOUBLE |  |  |
| `dispenses` | INTEGER |  |  |
| `total_cost` | DOUBLE |  |  |
| `days_supplied` | BIGINT |  |  |

### `fct_observations`  (core)
Observation fact (labs/vitals/survey). Grain: one row per measurement.

- **Relation:** `healthcare.main.fct_observations`
- **Primary key:** observation_id
- **Foreign keys:** patient_id → dim_patient.patient_id

| column | type | description | examples |
|---|---|---|---|
| `observation_id` | VARCHAR | Surrogate primary key. |  |
| `patient_id` | VARCHAR |  |  |
| `encounter_id` | VARCHAR |  |  |
| `observed_at` | TIMESTAMP |  |  |
| `observation_category` | VARCHAR |  | vital-signs, laboratory, survey |
| `observation_code` | VARCHAR |  | 55758-7, 5804-0, 6690-2 |
| `observation_description` | VARCHAR |  | Housing status, QOLS, Body Height |
| `value_text` | VARCHAR |  | 25.0, 121.4, English |
| `value_numeric` | DOUBLE |  |  |
| `units` | VARCHAR |  | {score}, s, pg/mL |
| `value_type` | VARCHAR |  | text, numeric |

### `fct_procedures`  (core)
Procedure fact. Grain: one row per procedure performed.

- **Relation:** `healthcare.main.fct_procedures`
- **Primary key:** procedure_event_id
- **Foreign keys:** patient_id → dim_patient.patient_id

| column | type | description | examples |
|---|---|---|---|
| `procedure_event_id` | VARCHAR | Surrogate primary key. |  |
| `patient_id` | VARCHAR |  |  |
| `procedure_code` | VARCHAR |  | 241046008, 52052004, 269911007 |
| `encounter_id` | VARCHAR |  |  |
| `procedure_start` | TIMESTAMP |  |  |
| `procedure_stop` | TIMESTAMP |  |  |
| `base_cost` | DOUBLE |  |  |
| `reason_code` | VARCHAR |  | 37320007, 267020005, 1231000119100 |
| `reason_description` | VARCHAR |  | Traumatic dislocation of temporomandibular joint (disorder), History of aortic valve replacement (situation), Proliferative diabetic retinopathy due to type II diabetes mellitus |

### `mart_condition_prevalence`  (analytics)
Condition prevalence by patient age band. One row per (condition_code, age_group). Small age bands make rare-condition estimates noisy — total_patients_in_age_group is the denominator, exposed so consumers can judge reliability.

- **Relation:** `healthcare.main.mart_condition_prevalence`
- **Primary key:** condition_code, age_group

| column | type | description | examples |
|---|---|---|---|
| `condition_code` | VARCHAR |  | 160904001, 73438004, 427419006 |
| `condition_description` | VARCHAR |  | Pneumonia (disorder), Proliferative diabetic retinopathy due to type II diabetes mellitus, Mitral valve regurgitation (disorder) |
| `age_group` | VARCHAR |  | 18-39, 0-17, 40-64 |
| `patients_with_condition` | BIGINT |  |  |
| `total_patients_in_age_group` | BIGINT |  |  |
| `prevalence_pct` | DOUBLE | 100 * patients_with_condition / total_patients_in_age_group. |  |

### `mart_cost_by_condition`  (analytics)
Cost of the diagnosing encounter aggregated by condition (NOT lifetime cost of care). One row per condition_code.

- **Relation:** `healthcare.main.mart_cost_by_condition`
- **Primary key:** condition_code
- **Foreign keys:** condition_code → dim_condition.condition_code

| column | type | description | examples |
|---|---|---|---|
| `condition_code` | VARCHAR | PK — SNOMED-CT condition code. | 88805009, 160904001, 1255252008 |
| `condition_description` | VARCHAR |  | Seizure disorder (disorder), Sputum finding (finding), Pneumonia (disorder) |
| `num_episodes` | BIGINT |  |  |
| `num_patients` | BIGINT | Distinct patients who have this condition. |  |
| `total_diagnosing_encounter_cost` | DOUBLE |  |  |
| `avg_diagnosing_encounter_cost` | DOUBLE |  |  |

### `mart_readmissions`  (analytics)
30-day inpatient readmission flags. One row per index inpatient encounter; is_30d_readmission is true when the same patient is admitted again within 0-30 days of discharge.

- **Relation:** `healthcare.main.mart_readmissions`
- **Primary key:** index_encounter_id
- **Foreign keys:** patient_id → dim_patient.patient_id

| column | type | description | examples |
|---|---|---|---|
| `index_encounter_id` | VARCHAR | PK — the index inpatient encounter. |  |
| `patient_id` | VARCHAR |  |  |
| `admission_date` | TIMESTAMP |  |  |
| `discharge_date` | TIMESTAMP |  |  |
| `next_admission_start` | TIMESTAMP |  |  |
| `days_to_next_admission` | BIGINT |  |  |
| `is_30d_readmission` | BOOLEAN | True if a subsequent inpatient admission occurred within 30 days of discharge. |  |
| `total_claim_cost` | DOUBLE |  |  |
