# Dataset Schema Inspection Notes

Date: 2026-05-26
Scope: temporary, non-persistent inspection of official Coswara and COUGHVID metadata/layout so the implementation matches real files instead of assumed schemas.

## Temporary Inspection Policy

Temporary downloads were placed under `/tmp/covid_audio_dataset_inspection` only. No dataset audio or raw metadata was copied into the persistent hidden project. Only code, tests, and this traceable inspection note were kept.

## Coswara Findings

Official sources inspected:

- Coswara GitHub repository root: `https://github.com/iiscleap/Coswara-Data`
- Combined metadata CSV: `https://raw.githubusercontent.com/iiscleap/Coswara-Data/master/combined_data.csv`
- Column legend JSON: `https://raw.githubusercontent.com/iiscleap/Coswara-Data/master/csv_labels_legend.json`
- Example date folder listing through GitHub API: `https://api.github.com/repos/iiscleap/Coswara-Data/contents/20200413`
- Example date metadata CSV: `https://raw.githubusercontent.com/iiscleap/Coswara-Data/master/20200413/20200413.csv`

Observed `combined_data.csv` shape during inspection: 2746 rows x 36 columns.

Observed columns:

```text
id, a, covid_status, record_date, ep, g, l_c, l_l, l_s, rU, smoker, cold, ht, diabetes,
cough, ctDate, ctScan, ctScore, diarrhoea, fever, loss_of_smell, mp, testType,
test_date, test_status, um, vacc, bd, others_resp, ftg, st, ihd, asthma,
others_preexist, cld, pneumonia
```

Important mappings from `csv_labels_legend.json`:

- `id`: user ID / participant ID
- `a`: age
- `g`: gender
- `l_c`: country
- `record_date`: date recorded/submitted
- `covid_status`: health status
- symptom-like fields include `cough`, `cold`, `fever`, `diarrhoea`, `loss_of_smell`, `bd`, `st`, `ftg`, `mp`, `others_resp`
- comorbidity/risk fields include `asthma`, `ht`, `diabetes`, `ihd`, `cld`, `pneumonia`, `smoker`, `others_preexist`

Observed repository layout:

- The repository has date folders such as `20200413`, `20200415`, etc.
- The inspected `20200413` folder contains `20200413.csv` plus split archives such as `20200413.tar.gz.aa`, `.ab`, `.ac`, `.ad`.

Implementation consequences:

- `data_index.build_participant_metadata` now recognizes official short Coswara columns (`a`, `g`, `l_c`, `record_date`, `testType`, `test_status`).
- `symptoms_json` and `comorbidities_json` are now built from the observed Coswara symptom/comorbidity fields.
- `build_audio_index` carries these fields forward so confounding checks and metadata baselines are possible.
- A regression test now checks the official short-column path.

## COUGHVID Findings

Official sources inspected:

- COUGHVID Zenodo record JSON inspected for record `4048312`: `https://zenodo.org/api/records/4048312`
- COUGHVID `public_dataset.zip` from that record was temporarily downloaded and inspected with `unzip -l` / `unzip -p`.
- The source registry also tracks later/current COUGHVID dataset references such as Zenodo record `7024894` and the Scientific Data article. The adapter supports both sidecar and CSV-style metadata paths.

Observed record `4048312` file list:

- The record exposes `public_dataset.zip` as the main dataset archive.
- No separate `metadata_compiled.csv` was observed in that record listing.

Observed zip layout:

```text
public_dataset/<uuid>.json
public_dataset/<uuid>.webm
public_dataset/<uuid>.ogg
```

Example sidecar JSON fields inspected:

```json
{
  "datetime": "2020-04-13T21:30:59.801831+00:00",
  "cough_detected": "0.9609",
  "latitude": "31.3",
  "longitude": "34.8",
  "age": "15",
  "gender": "male",
  "respiratory_condition": "False",
  "fever_muscle_pain": "False",
  "status": "healthy"
}
```

Implementation consequences:

- `external_datasets.build_coughvid_index` now supports extracted sidecar layout, direct `public_dataset.zip` inspection, and CSV metadata such as `metadata_compiled.csv`.
- Sidecar status values are mapped into the project label vocabulary: `healthy` -> negative, COVID-like labels -> positive, ambiguous respiratory/symptomatic labels -> unknown.
- `cough_detected` is retained as a manual quality score and converted to `ok` / `uncertain` / `bad` quality labels.
- Direct zip indexing writes audio paths as `archive.zip::public_dataset/<uuid>.webm` or `.ogg`.
- `audio_io.load_audio` can now materialize `archive.zip::member` paths into temporary files for normal `soundfile` / `librosa` loading.
- Regression tests cover sidecar directory layout, direct zip layout, v3-style `status_SSL` CSV metadata, and temporary zip-member materialization.

## Remaining Real-Data Validation

These changes make the schema handling realistic, but they do not replace a full local run:

- Coswara split archives still need to be extracted on the local machine and passed through the layout audit.
- COUGHVID direct zip loading is supported, but extracting the archive once is still recommended for speed during bulk feature extraction.
- Runtime tests were not run on this EC2 image because `numpy` and `pytest` are missing from the system environment.
- The publication notebook should be run after dependencies are installed locally and Coswara/COUGHVID paths are available.
