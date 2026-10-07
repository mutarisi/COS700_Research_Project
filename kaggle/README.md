# Running the training-seed experiment on Kaggle

This folder holds everything needed to run the training-seed robustness
experiment on Kaggle instead of on the laptop.

| File | What it is |
|---|---|
| `training_seed_experiment.ipynb` | The notebook to run. It contains the project's code, so nothing else from `src/` needs uploading |
| `DSL-StrongPasswordData.csv` | The CMU dataset the notebook needs as an input (a copy of `data/raw/`) |
| `build_notebook.py` | Rebuilds the notebook from `src/` if the code changes. Not needed on Kaggle |

## What the run does

It retrains every subject's Isolation Forest and LSTM Autoencoder under five
training seeds (42, 1, 2, 3, 4) and evaluates each at 18 sampling resolutions
x 10 phase-jitter seeds, for all 51 subjects. Expect roughly 3 hours on a
Kaggle CPU session. No GPU and no internet access are needed.

## Steps

Kaggle's button labels change from time to time; the names below are the
usual ones.

1. Sign in at kaggle.com and choose **Create -> New Notebook**.
2. In the notebook, choose **File -> Import Notebook**, and upload
   `training_seed_experiment.ipynb` from this folder.
3. In the right-hand panel choose **Add Input -> Upload**, upload
   `DSL-StrongPasswordData.csv` as a new dataset (any name; keep it
   **private**), and wait until it shows under Input.
4. In **Settings / Session options**, leave the accelerator on **None**.
5. Click **Save Version**, choose **Save & Run All (Commit)**, and save.
   The run now continues on Kaggle's servers; the browser and the laptop can
   be closed.
6. When the version shows as complete, open it, go to the **Output** tab and
   download `training_seeds_eer_by_subject.csv` (about 8 MB).
7. Put that file in the project's `results/` folder.

## If something goes wrong

- **"DSL-StrongPasswordData.csv not found"**: the dataset from step 3 is not
  attached to the notebook. Add it under Input and run again.
- **"Trial run failed"**: the two-minute trial in section 4 of the notebook
  did not produce valid results. Copy the output of that cell; it shows what
  broke.
- **"Run is incomplete"**: the session stopped early. Open the notebook in
  the editor and choose Run All again in the same session; finished subjects
  are skipped. A fresh session starts from the beginning, since Kaggle does
  not keep working files between sessions.

## What to send back

`training_seeds_eer_by_subject.csv`, plus the line of library versions printed
in section 3 of the notebook (Python, TensorFlow, scikit-learn, NumPy, pandas).
Kaggle's library versions differ from the laptop's, so the seed-42 numbers
from Kaggle will not match the laptop's exactly; that is expected, and all
five seeds are compared within the Kaggle run.

## The run used for the report

Completed on a Kaggle CPU session with Python 3.13.15, TensorFlow 2.20.0, scikit-learn 1.6.1,
NumPy 2.1.3 and pandas 2.3.3. Its output is `results/training_seeds_eer_by_subject.csv`
(130,815 rows). The seed-42 results match the laptop's run (Python 3.11.0, TensorFlow 2.21.0,
scikit-learn 1.9.0, NumPy 2.4.6, pandas 3.0.3) exactly for Isolation Forest and to within
0.01 EER points on average for the LSTM Autoencoder.
