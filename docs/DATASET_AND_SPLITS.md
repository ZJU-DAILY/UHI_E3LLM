# Dataset lineage and splits

## Arrow time-series layout

Each city dataset stores one Arrow record with a `year_month` vector and a
`values` tensor whose shape is `[months, 24]`. Channels follow the order in
`scripts/prepare_timeseries_baselines.py`: 13 structural attributes, 10 monthly
environmental variables, and UHI. The train tensor covers 2003-01 to 2017-12;
the validation tensor covers 2018-01 to 2020-12. A 12-month input history and
the target-month UHI are sliced from the sequence when a training or evaluation
sample is read.

## City selection

```text
approximately 6,000 initially collected cities
  -> 1,020 eligible cities after data-completeness and area filtering
  -> 600 modeling cities selected by stratified random sampling
     across the five city-size categories
       -> 500 training cities
       -> 100 held-out test cities
  -> 100 independent calibration cities sampled from the remaining candidates
```

The released CSV manifests define the cohorts used in the reported experiments.
Use them with the processed dataset for training and evaluation.
`scripts/sample_modeling_cities.py` constructs a new stratified city selection
when rebuilding the dataset.

The city-size allocation in the released CSV manifests is:

| Stratum | Eligible (1,020) | Modeling (600) | Train (500) | Test (100) |
|---|---:|---:|---:|---:|
| Very small | 336 | 192 | 159 | 33 |
| Small | 462 | 277 | 229 | 48 |
| Medium | 105 | 54 | 44 | 10 |
| Large | 61 | 43 | 39 | 4 |
| Very large | 56 | 34 | 29 | 5 |

## Temporal splits

| Role | Cities | Target period | Effective samples |
|---|---:|---|---:|
| SFT/PPO training | 500 | 2004-01 to 2017-12 | 84,000 |
| Validation/model selection | 500 | 2018-01 to 2020-12 | 18,000 |
| Held-out test | 100 | 2018-01 to 2020-12 | 3,600 |
| Independent calibration | 100 | Defined by calibration workflow | Recorded by workflow |

## Area strata

The five area strata group cities for size-category reporting:

| Stratum | Area range (km2) |
|---|---:|
| Very small | 100-150 |
| Small | 150-450 |
| Medium | 450-800 |
| Large | 800-1500 |
| Very large | >=1500 |

## Regional folds

The regional folds use fixed WGS84 city-center longitude and latitude boundaries
implemented in `scripts/assign_city_regions.py`. All methods use the same
released fold manifests. `SCWAsia` is the filename key for the partition labelled
West Asia in the manuscript. The boundaries are:

- longitude below -30 degrees: North America at latitude 10 degrees or above,
  otherwise Latin America;
- longitude from -30 to below 55 degrees and latitude below 35 degrees: Africa;
- longitude from -30 to below 40 degrees and latitude 35 degrees or above: Europe;
- longitude 93 degrees or above: East Asia & Oceania;
- all remaining coordinates: West Asia.
