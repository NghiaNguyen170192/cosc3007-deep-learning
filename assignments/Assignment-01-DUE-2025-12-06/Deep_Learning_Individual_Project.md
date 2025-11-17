# Deep Learning Individual Project — Assignment Specification (Markdown)

**Due date: 11:59 PM, Saturday, December 6, 2025**

---

## 1. Overview
You are provided with a custom dataset containing **16,000 samples** from **4 classes**.  
Your task is to design, train, justify, and submit a **deep learning classifier** using **Keras / TensorFlow**.  
This is an **individual** project. You must submit both code and a trained model.  
You will be assessed on correctness, reasoning, experimentation quality, clarity, and reproducibility.

---

## 2. Files You Will Receive
- `4class_32x32.npz`  
  - `X`: `(16000, 32, 32)` `float32`  
  - `y`: `(16000,)` `int64` with labels in `{0,1,2,3}`  
- There is **no predefined train/validation split**. You must create your own split(s).

---

## 3. What You Must Submit
- `submission_studentId.ipynb`  (e.g., `submission_s3938294.ipynb`)  
- `model_studentId.h5`          (e.g., `model_s3938294.h5`)

Your notebook must support **both** options:

| Option | Description |
|-------|-------------|
| **A. Load Model** | Loads `model_studentId.h5` and evaluates |
| **B. Train Model** | Full training pipeline from scratch |

Both options must produce the same callable:

```python
def predict_fn(X32x32: np.ndarray) -> np.ndarray:
    ...
```
- Input: `(N, 32, 32)` float32 arrays
- Output: `(N,)` `int64` labels in `{0,1,2,3}`

---

## 4. Notebook Requirements
Your notebook must include and clearly separate the following sections:

- **Introduction (markdown)**: objective and overview  
- **Dataset inspection (markdown + code)**: value ranges, comments  
- **Train/validation split & preprocessing (markdown)**: choices justified  
- **Architecture reasoning (markdown)**: why this model and layer sizes  
- **Techniques & theory (markdown)**: activation, loss, optimizer, regularization  
- **Experiments (markdown + code)**: variations tried, learning curves, ablations  
- **Option A — Load trained model**: `keras.models.load_model(...)`  
- **Option B — Train from scratch**: reproducible code; save with `model.save("model_studentId.h5")`  
- **`predict_fn(X32x32)` definition (mandatory)**: works after either path  
- **Dev-set evaluation and discussion**  
- **Reflection**: weaknesses, improvements, insights  

---

## 5. Allowed Libraries and Environment Rules
- **Framework**: Keras / TensorFlow only (no PyTorch/JAX/etc.).  
- **Allowed libraries**: `numpy`, `matplotlib`, `pandas`, `scikit-learn`, `seaborn`, and similar standard scientific Python packages.  
- **Internet is allowed.** The notebook must run **without the instructor manually installing packages**. Lightweight `pip install` **inside** the notebook is permitted if necessary and clearly stated.  
- **Pretrained models** are allowed if you:
  1. Clearly cite the source (e.g., `keras.applications.*`),  
  2. Explain why the choice is appropriate,  
  3. Ensure your notebook still runs without manual setup.  
- **Do not download** any external datasets during execution.

---

## 6. Academic Honesty Statement
Include this text at the top of your notebook:

> I declare that this submission is my own work, and that I did not use any pretrained model or code that I did not explicitly cite.

---

## 7. Grading Notes

| Criterion | Weight |
|-----------|--------|
| **Overall structure, code quality & reproducibility** | **10%** |
| **Depth of reasoning / theoretical justification** | **10%** |
| **Correctness of implementation & model accuracy** | **50%** |
| **Documentation: quality of experiments & supporting evidence** | **30%** |

There is **no strict accuracy threshold**, but poor performance without justification will affect the mark.

---

## 8. Instructor Evaluation Workflow
1. Open `submission_studentId.ipynb`  
2. Run **Option A** to load `model_studentId.h5`  
3. Call `predict_fn(X)` on a **hidden hold-out dataset**  
4. Review markdown reasoning, experiments, and code clarity  
5. If needed, run **Option B** to verify training integrity

---

## 9. Submission Format
Submit exactly **two files**, both including your student ID:

| File | Format | Example |
|------|--------|---------|
| Notebook | `submission_studentId.ipynb` | `submission_s3938294.ipynb` |
| Model    | `model_studentId.h5`         | `model_s3938294.h5`         |

Do **not** zip unless specifically requested. Late submissions follow standard policy.
