# TP03 - Numerical Methods and Optimization (NM&O) - First Semester 2024

This practical assignment focuses on the analysis and application of **Singular Value Decomposition (SVD)** and its use in data compression, dimensionality reduction, and linear regression tasks.

📄 [Download TP03 Report in PDF](MNyO_TP03.pdf.zip)

---

## 🖼️ PART 1: Image Compression

A compression technique based on **SVD** is implemented on a set of images represented as vectors of size \(p \times p\).

### Objectives:

- Apply **SVD** to learn a low-dimensional representation  
- Visualize reconstructed images using different numbers of principal components  
- Analyze compression error (Frobenius norm) as a function of the dimension \(d\)  
- Determine the value of \(d\) that ensures an error below 10%

---

## 📉 PART 2: Dimensionality Reduction and Least Squares

This part uses a dataset of sensor measurements (`dataset.csv`) and a response variable (`y.txt`) to perform dimensionality reduction and linear prediction.

### Subsections:

#### 🔻 2.1 Dimensionality Reduction

- **SVD** is applied to reduce the original dataset \(X\) to a new space \(Z = V_d^\top X\)  
- Pairwise similarities between samples are analyzed in spaces of different dimensions (d = 2, 6, 10, p) using **PCA**  
- Similarity matrices are visualized and the optimal value of \(d\) is discussed based on the structure of the singular values

#### 📐 2.2 Least Squares Regression

- A regression model \( \hat{y} = X \hat{\beta} \) is trained by solving the least squares problem in the original space  
- The weights assigned to each original dimension in the vector \( \hat{\beta} \) are analyzed

#### 🔁 2.3 Regression in Reduced Space

- A model is trained in the reduced space \(Z\) with \(d = 2\)  
- Prediction error is compared with the original model, and the best-fit samples are identified

---

## ✅ Requirements

- Python 3.x  
- numpy  
- matplotlib  
- pandas  
- scikit-learn  
- Jupyter Notebook
