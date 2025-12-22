# Predicting Player Performance in Fantasy Football (Serie A) using Deep Learning

**Course:** Foundations of Data Science  
**Authors:** Francesco Finazzi, Chiara Vulpiani

## Project Overview

This project models "Fantacalcio" (Italian Fantasy Football) score prediction as a time-series regression problem. In Fantacalcio, participants manage virtual teams of real Serie-A players, where scores ("Fantavoto") combine subjective journalist grades with objective event-based bonuses and maluses.

Accurate prediction is critical for solving the weekly "sit/bench" dilemma. Unlike previous works focusing on the statistics-driven English Premier League, this project addresses the subjective nature of Serie A ratings using Deep Learning, building on insights regarding extensive feature engineering and recent form patterns.

## Dataset Construction

We constructed a comprehensive dataset spanning the **2017-18 to 2024-25 Serie A seasons**.

* **Data Sources:**
    * **Match Statistics:** Scraped from FBref.com using a modified scraper to bypass bot detection.
    * **Subjective Grades:** Scraped from Fantacalcio.it via custom scripts.
* **Preprocessing:**
    * Player identities aligned via fuzzy string matching.
    * **Exclusions:** The 2020-21 season (COVID anomalies) and Goalkeepers (different statistical profiles) were removed.
    * **Final Size:** 68,708 player-match observations covering 1,552 unique players.
* **Data Splitting:** Chronological split to prevent data leakage.
    * **Training:** Seasons 2017-2023 (N ≈ 8,744).
    * **Validation:** Season 2023-24 (N ≈ 8,476).
    * **Test:** Season 2024-25 (N ≈ 8,391).

## Methodology & Architecture

### 1. Feature Engineering
From an initial 78 raw features, we reduced dimensionality through filtering, collinearity reduction (removing r > 0.75), and Permutation Feature Importance. The final input vector consists of:
* **30 Player Features:** Historical stats.
* **6 Team Features:** Historical team stats.
* **4 Future Context Features:** Data regarding the upcoming match.

> **Note:** A Robust MinMax scaler was used to handle outliers.

### 2. Auxiliary XGBoost Model
Feature importance analysis revealed that future context dominates prediction. To provide this signal explicitly, we trained an **XGBoost Regressor** to predict the specific outcome of the upcoming match (Team Goals Scored/Conceded).

### 3. Dual-Input Neural Network
We designed a modular dual-stream architecture:
1.  **Context Stream (Future):** Processes static future context (including XGBoost predictions) via Dense Layers.
2.  **Temporal Stream (History):** Processes historical tensor using either:
    * **LSTM Variant:** Captures sequential trends.
    * **CNN Variant:** Captures local patterns via stacked 1D Convolutional layers.

These streams are concatenated and passed through an MLP head to predict the "Fantavoto".

### 4. Hybrid Loss Function
To balance error minimization (MSE) and ranking accuracy (Cosine Similarity), we utilized a hybrid loss function:

$$Loss = \alpha MSE + (1 - CosineSimilarity)$$

This addresses the tendency of MAE to be too conservative and MSE to penalize outliers excessively.

## Experimental Results

Models were evaluated using Mean Absolute Error (MAE) and Spearman Correlation Coefficient (ρ).

### Performance Comparison (All Players)
The inclusion of XGBoost context led to a **57% improvement in ρ** over the baseline.

| Loss Used | Model | MAE | Spearman (ρ) |
| :--- | :--- | :--- | :--- |
| **Baseline** | Linear Regression | 0.8264 | 0.1813 |
| **MAE Loss** | LSTM Base | 0.7375 | 0.1059 |
| | LSTM + XGBoost | 0.7187 | 0.2587 |
| | CNN + XGBoost | **0.7162** | 0.2551 |
| **Hybrid Loss**| LSTM + XGBoost | 0.7797 | 0.2801 |
| | CNN + XGBoost | 0.7840 | 0.2774 |
| **Ensemble** | LSTM (MAE) + CNN (HL) | 0.7179 | 0.2581 |
| **Ensemble** | LSTM (HL) + CNN (MAE) | 0.7477 | **0.2852** |

### Results by Role
Defenders are the most predictable due to performance stability, while attackers are highly volatile due to reliance on rare events (goals).

| Role | Best Result MAE | Best Result Spearman (ρ) |
| :--- | :--- | :--- |
| Defenders | 0.5814 | 0.3105 |
| Midfielders | 0.6890 | 0.2774 |
| Attackers | 1.030 | 0.2584 |
| **Total** | **0.7162** | **0.2851** |

### Key Findings
* **Context is Key:** Team performance context is essential for individual grade prediction.
* **Error-Ranking Tradeoff:** Models trained on MAE produced stable/conservative predictions, while Custom Loss models had higher absolute error (~9% increase) but superior ranking accuracy (~8% increase in ρ).
* **Ensembling:** Combining MAE-trained and Hybrid-trained models offers the best balance for practical application.

## Future Work
1.  **Sentiment Analysis:** Analyze sports journalism to capture qualitative factors.
2.  **Specialized Models:** Develop dedicated models for goalkeepers and "high-variance" probability models for attackers.
