"""
Builds kaggle/training_seed_experiment.ipynb from the project's own source
files, so the notebook always runs exactly the code in src/.

The notebook is self-contained: it writes the needed src/*.py files itself,
so the only thing to upload to Kaggle besides the notebook is the CMU CSV.

Run from the project root after changing any of the embedded source files:
    python kaggle/build_notebook.py
"""
import os
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "kaggle", "training_seed_experiment.ipynb")

# the experiment script and everything it imports
EMBEDDED = ["eer.py", "downsample.py", "subject_models.py",
            "run_multiseed_experiment.py", "run_training_seeds.py"]

cells = [
    new_markdown_cell(
        "# Training-seed robustness experiment\n\n"
        "Retrains every subject's Isolation Forest and LSTM Autoencoder under several training seeds and "
        "re-evaluates them at 18 sampling resolutions x 10 phase-jitter seeds (51 subjects).\n\n"
        "**How to run:** add `DSL-StrongPasswordData.csv` as an input, then use "
        "**Save Version -> Save & Run All (Commit)**. It takes roughly 3 hours on a CPU session and keeps "
        "running with the browser closed.\n\n"
        "**What to bring back:** `training_seeds_eer_by_subject.csv` from the Output tab."),
    new_markdown_cell("## 1. Settings\nLeave these as they are for the real run."),
    new_code_cell(
        "TRAIN_SEEDS = [42, 1, 2, 3, 4]   # model training seeds\n"
        "N_JITTER_SEEDS = 10              # phase-jitter seeds per downsampled resolution\n"
        "N_SUBJECTS = None                # None = all 51 subjects"),
    new_markdown_cell("## 2. Project source files\nThese cells write the project's code into `src/`."),
    new_code_cell("import os\nfor d in ['src', 'data/raw', 'results', 'trial']:\n    os.makedirs(d, exist_ok=True)"),
]

for name in EMBEDDED:
    source = open(os.path.join(ROOT, "src", name), encoding="utf-8").read().rstrip("\n")
    cells.append(new_code_cell(f"%%writefile src/{name}\n{source}"))

cells += [
    new_markdown_cell("## 3. Find the dataset"),
    new_code_cell(
        "import glob, shutil\n"
        "import pandas as pd\n\n"
        "NAME = 'DSL-StrongPasswordData.csv'\n"
        "candidates = (glob.glob(f'/kaggle/input/**/{NAME}', recursive=True)\n"
        "              + glob.glob(NAME) + glob.glob(f'../data/raw/{NAME}'))\n"
        "assert candidates, (f'{NAME} not found. In the right-hand panel choose Add Input -> Upload, '\n"
        "                    'and upload the CSV as a new dataset, then run again.')\n"
        "shutil.copy(candidates[0], f'data/raw/{NAME}')\n"
        "df = pd.read_csv(f'data/raw/{NAME}')\n"
        "print('Using', candidates[0])\n"
        "print(df.shape, '-', df['subject'].nunique(), 'subjects')\n"
        "assert df.shape == (20400, 34) and df['subject'].nunique() == 51, 'This is not the expected CMU file'"),
    new_code_cell(
        "import sys, sklearn, numpy, tensorflow as tf\n"
        "print('python', sys.version.split()[0], '| tensorflow', tf.__version__, '| scikit-learn', sklearn.__version__,\n"
        "      '| numpy', numpy.__version__, '| pandas', pd.__version__)\n"
        "print('GPU:', tf.config.list_physical_devices('GPU') or 'none (CPU run)')"),
    new_markdown_cell(
        "## 4. Two-minute trial\nRuns 2 subjects with 1 training seed, to confirm everything works here "
        "before the long run starts."),
    new_code_cell(
        "!{sys.executable} src/run_training_seeds.py --n-subjects 2 --n-seeds 2 --train-seeds 42 "
        "--out-dir trial --out trial/trial.csv 2>&1 | grep -v -i -E \"warning|oneDNN|tensorflow/core|^W0000|^I0000|^E0000\"\n"
        "trial = pd.read_csv('trial/trial.csv')\n"
        "assert len(trial) == 2 * 35 * 3 and trial['eer'].between(0, 1).all(), 'Trial run failed'\n"
        "print('Trial OK:', len(trial), 'rows')"),
    new_markdown_cell(
        "## 5. Full run\nAbout 35-45 seconds per subject per training seed. Progress is saved after every "
        "subject, so re-running this cell continues where it stopped."),
    new_code_cell(
        "seeds = ' '.join(str(s) for s in TRAIN_SEEDS)\n"
        "subjects = f'--n-subjects {N_SUBJECTS}' if N_SUBJECTS else ''\n"
        "!{sys.executable} src/run_training_seeds.py --train-seeds {seeds} --n-seeds {N_JITTER_SEEDS} {subjects} "
        "2>&1 | grep -v -i -E \"warning|oneDNN|tensorflow/core|^W0000|^I0000|^E0000\""),
    new_markdown_cell("## 6. Check and package the result"),
    new_code_cell(
        "result = pd.read_csv('results/training_seeds_eer_by_subject.csv')\n"
        "n_subjects = N_SUBJECTS or 51\n"
        "n_conditions = 1 + 17 * N_JITTER_SEEDS          # 1 ms control + 17 downsampled resolutions\n"
        "expected = len(TRAIN_SEEDS) * n_subjects * n_conditions * 3\n"
        "print(len(result), 'rows; expected', expected)\n"
        "assert len(result) == expected, 'Run is incomplete - re-run the Full run cell to continue'\n"
        "assert result['eer'].notna().all()\n"
        "shutil.copy('results/training_seeds_eer_by_subject.csv', 'training_seeds_eer_by_subject.csv')\n"
        "print('\\nDONE. Download training_seeds_eer_by_subject.csv from the Output tab.')"),
]

nb = new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
})
nbformat.validate(nb)
nbformat.write(nb, OUT)
print(f"Wrote {OUT} ({len(cells)} cells, {os.path.getsize(OUT) / 1024:.0f} KB)")
