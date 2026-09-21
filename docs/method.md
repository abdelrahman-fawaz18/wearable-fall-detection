# Methodology

## Acquisition

The wrist-worn unit combines a Raspberry Pi Zero W and an MPU9250 inertial measurement unit. Motion is recorded at 100 Hz with these sensor settings:

| Sensor | Configuration |
|---|---|
| Accelerometer | ±8 g |
| Gyroscope | ±1000 °/s |
| Magnetometer | 16-bit output, 100 Hz continuous mode |

The acquisition program writes UTC timestamps and nine motion channels to CSV.

## Frame preparation

Adjacent 100 Hz readings are averaged to produce a 50 Hz signal:

\[
x_{50}[n] = \frac{x_{100}[2n] + x_{100}[2n+1]}{2}
\]

Dynamic recordings are standardized to 50 samples. Recordings longer than one second contribute the contiguous window with the highest acceleration-vector energy. Shorter recordings are extended by reflecting their trailing samples. Static recordings are divided into consecutive one-second frames.

The repository implementation is in [`preprocessing.py`](../src/wearable_fall_detection/preprocessing.py).

## Feature extraction

The analysis code uses the first 46 samples of each prepared frame and calculates Pearson autocorrelation for lags 1 through 14 independently on all six channels:

\[
r_k = \operatorname{corr}(x_t, x_{t-k}), \qquad k \in \{1, \ldots, 14\}
\]

The result is an 84-element vector ordered by lag, then channel.

## Classifiers

Both classification tasks use an RBF support-vector classifier with `C=100` and `gamma=0.01`.

### Multiclass

The multiclass task distinguishes eight activities: fall, downstairs, upstairs, walking, jogging, sitting, standing, and lying.

### Binary

The binary task maps every daily activity to `No fall` and retains `Fall` as the positive event.

## Evaluation protocol

Repository metrics are the arithmetic mean across ten 80/20 holdout evaluations with split seeds 0–9. Each split is stratified by its task labels. Precision, recall, and F1 are weighted by class support. Aggregate confusion matrices sum the test predictions from all ten runs and display row-normalized proportions.

The evaluation code writes:

- `benchmark-summary.json` — mean metrics by task;
- `benchmark-runs.csv` — metrics for every split;
- `confusion-multiclass.csv` and `confusion-binary.csv` — aggregate counts;
- one full-dataset model per task.
