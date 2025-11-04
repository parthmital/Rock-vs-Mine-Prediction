# Rock vs Mine Prediction using Machine Learning

## Overview
This project uses **Logistic Regression**, a supervised learning algorithm, to classify sonar signals as either **Rock (R)** or **Mine (M)** based on frequency data.  
The dataset consists of 60 numerical features representing sonar echo intensities at various angles.  
The goal is to train a binary classifier that distinguishes between metal (mines) and rock objects beneath the sea.

---

## Dataset Information
**File:** `sonar_data.csv`  
**Source:** UCI Machine Learning Repository  
**Details:**
- 208 samples
- 60 continuous features (sonar frequency readings)
- 1 label column (`R` = Rock, `M` = Mine)

Each instance is a set of sonar readings bounced off a surface — the reflections differ depending on whether the surface is metallic (mine) or rocky.
