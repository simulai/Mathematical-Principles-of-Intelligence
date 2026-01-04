# Kaggle Competition Guide for MPI Project

This guide details how to run the MPI (Softmax-Projected Hyper-Attention) model on Kaggle, specifically for the **Natural Language Processing with Disaster Tweets** competition.

## 1. Quick Start (Kaggle Notebook)

This is the easiest method and the one verified to work with the current codebase.

### Step 1: Create a Notebook
1. Go to the [Disaster Tweets Competition Page](https://www.kaggle.com/competitions/nlp-getting-started).
2. Click **Code** -> **New Notebook**.

### Step 2: Add Data
The default competition data (`train.csv`, `test.csv`) is usually added automatically.
**If you have your own preprocessed data (like `train_preprocessed.csv`):**
1. In the right sidebar, find the **Input** section.
2. Click **Upload** > **Dataset**.
3. Drag and drop your `.csv` files.
4. Give it a title (e.g., `mpi-disaster-tweets-data`) and click **Create**.
5. Once uploaded, the files will appear under `/kaggle/input/mpi-disaster-tweets-data/` (or similar).

### Step 3: Copy Code
1. Open the file `notebooks/kaggle_source_code.py` in your local project.
2. **Copy the entire content** of this file.
3. Paste it into the first code cell of your Kaggle Notebook.

### Step 4: Run Training
1. Ensure your accelerator is set to **GPU T4 x2** (or P100) in the right sidebar under **Session options**.
2. Click **Run All** or execute the cell.
3. The script will:
   - Automatically detect your data files (even if named `text_clean` or `text_cleaned`).
   - Train the DistilBERT + SPHA + CognitiveHolonomy model.
   - Generate `submission.csv`.

### Step 5: Submit Predictions

**Option A: Manual Upload (If "Submit" button is missing)**
1. In the Right Sidebar -> Output section, click **Download** next to `submission.csv`.
2. Go to the [Competition Submission Page](https://www.kaggle.com/c/nlp-getting-started/submit).
3. Drag and drop the downloaded `submission.csv` file there.
4. **Description**: See "Submission Descriptions" below.
5. Click **Make Submission**.

### Suggested Submission Descriptions

**Option 1: Professional (Recommended)**
> DistilBERT + MPI Architecture (SPHA Attention + Cognitive Holonomy Loss). Implements e-base scaling law (b=e) for optimal information flow.

**Option 2: Short**
> MPI-DistilBERT with SPHA and Ricci Flow Regularization.

**Option 3: Detailed**
> Backbone: DistilBERT-base-uncased. 
> Head: Softmax-Projected Hyper-Attention (SPHA) with num_heads=8, branching_factor=e. 
> Loss: CrossEntropy + Cognitive Holonomy (lambda=0.05) for topological smoothing.

**Option B: Save & Commit (Standard)**
1. Click the **Save Version** button in the top right corner.
2. Select **Save & Run All (Commit)**.
3. Wait for the background run to finish.
4. Click on the version number or go back to the Notebook Viewer page (exit the editor).
5. Scroll down to the **Output** section.
6. Click **Submit** next to `submission.csv`.

## 2. Model Architecture Details

The code uses a specialized architecture designed for efficient attention:

- **Backbone**: `distilbert-base-uncased` (Lightweight, fast).
- **MPI Attention (SPHA)**:
  - **Heads**: 8
  - **Branching Factor**: $e$ (approx 2.718)
  - **Scaling**: $\frac{\ln(b)}{b}$ based on E-Base Scaling Law.
- **Regularization**: **Cognitive Holonomy Loss** ($\lambda_h=0.05$) to ensure smooth metric evolution (Ricci flow inspired).

## 3. Troubleshooting

### "Dataset not found"
- The script searches standard paths like `/kaggle/input/nlp-getting-started`.
- If you uploaded custom data, it searches recursively in `/kaggle/input`.
- **Fix**: Ensure your files are named `train.csv`/`test.csv` OR `train_preprocessed.csv`/`test_preprocessed.csv`.

### "KeyError: 'text'"
- This happens if your CSV column names don't match.
- **Fix**: The script now supports `text`, `text_cleaned`, `clean_text`, `tweet`, `content`. Ensure your text column matches one of these.

### "Out of Memory" (OOM)
- If using a larger backbone (e.g., BERT-Large), reduce `batch_size` in the `DataLoader` (default is 32 or 16).
