# Dataset

## Prepared frames

`data/frames` contains 1,191 headerless CSV files. Each file is one second of wrist motion sampled at 50 Hz, represented as a 50 × 6 matrix.

| Column | Signal | Unit |
|---:|---|---|
| 1 | Acceleration X | g |
| 2 | Acceleration Y | g |
| 3 | Acceleration Z | g |
| 4 | Angular velocity X | °/s |
| 5 | Angular velocity Y | °/s |
| 6 | Angular velocity Z | °/s |

The MPU9250 acquisition program records all nine inertial channels. The prepared model frames retain the accelerometer and gyroscope channels used by the repository classifiers.

## Filename mapping

The class is encoded by the filename prefix.

| Prefix | Activity | Files |
|---|---|---:|
| `Fall` | Fall | 165 |
| `dws` | Downstairs | 151 |
| `ups` | Upstairs | 150 |
| `wlk` | Walking | 150 |
| `jog` | Jogging | 150 |
| `sit` | Sitting | 150 |
| `std` | Standing | 150 |
| `lie` | Lying | 125 |

`data/dataset-manifest.csv` provides one row per file with its label and validated dimensions.

## Activity sources

Fall recordings were collected with the wrist-worn device across forward, backward, left-lateral, right-lateral, and vertical fall motions. Thirty participants contributed to the collection described in the publication.

Daily-activity recordings originate from the public activity dataset used in the study:

I. M. Pires, F. Hussain, G. Marques, and N. M. Garcia, “Comparison of machine learning techniques for the identification of human activities from inertial sensors available in a mobile device after the application of data imputation techniques,” *Data in Brief*, vol. 32, 2020. [https://doi.org/10.1016/j.dib.2020.106628](https://doi.org/10.1016/j.dib.2020.106628)

## Integrity checks

`fall-detection validate` verifies:

- all filenames map to a defined activity;
- every file has exactly 50 rows and six columns;
- every value is numeric and finite;
- the manifest matches the directory contents.
