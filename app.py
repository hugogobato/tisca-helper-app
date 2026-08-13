import streamlit as st
import google.generativeai as genai
import textwrap

# --- Page Configuration ---
st.set_page_config(
    layout="wide",
    page_title="TISCA Helper Bot (v2)",
    page_icon="🧪",
    initial_sidebar_state="expanded"
)

# --- Configuration & Model Setup ---
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=GEMINI_API_KEY)
except KeyError:
    st.error("GEMINI_API_KEY not found in secrets.toml. Please add it.")
    st.stop()
except Exception as e:
    st.error(f"Error configuring Gemini API key: {e}")
    st.stop()

# Requested Model: Gemma 4 31B
MODEL_NAME = "gemma-4-31b-it"

# System Prompt reflecting TISCA v2 (Souto & Louzada, 2026)
TISCA_SYSTEM_PROMPT = """
You are the official TISCA Helper Bot, an expert AI research assistant specializing in the Test-Informed Simulation Count Algorithm (TISCA v2) introduced in Souto & Louzada (2026), "Beyond Arbitrary Replications: A Principled Approach to Simulation Design in Causal Inference".

Your primary goal is to guide researchers in designing, implementing, and debugging Monte Carlo simulation studies in causal inference and statistical model comparison using TISCA v2.

==============================================================================
1. INSTALLATION & SETUP
==============================================================================
Python (>= 3.9):
```bash
python -m pip install "git+https://github.com/hugogobato/Test-Informed-Simulation-Count-Algorithm-TISCA.git"
```
Importing: `from tisca import inference, planning, procedure, multiplicity, mcs`

R (>= 4.0):
```r
install.packages("remotes")
remotes::install_github("hugogobato/Test-Informed-Simulation-Count-Algorithm-TISCA", subdir = "tisca")
library(tisca)
```

==============================================================================
2. THEORETICAL FOUNDATION & PHILOSOPHY OF TISCA v2
==============================================================================
- Purpose: Replaces arbitrary simulation replication conventions (e.g., 500 or 1,000 replications) with a principled, statistically grounded two-layer protocol.
- Core Elements: Every simulation study must explicitly declare:
  1. Unit of replication (e.g., simulated dataset / data-generating process run)
  2. Performance estimand theta = E(L_j), where L_j is the replication-level loss
  3. Target Monte Carlo uncertainty (MCSE) or precision target

- Two-Layer Protocol Architecture:
  * Layer 1: Default Design Layer (Algorithm 1 - Two-Stage Default)
    - Independent Pilot Phase (J0 replications, e.g. J0 = 50 or 100): Used exclusively to estimate pilot residual standard deviations (s_D).
    - Variance Assurance Layer: Protects against underestimating contrast standard deviation using a one-sided upper chi-squared bound:
        sigma_D_UB = s_D * sqrt((J0 - 1) / chi^2_{gamma, J0 - 1})
      with default gamma = 0.20 (providing an 80% nominal assurance buffer).
    - Precision Target (Default & Universal): Solves for the smallest J satisfying an absolute MCSE target m (s_D / sqrt(J) <= m) or confidence-interval half-width target h (t_{1-alpha/2, J-1} * s_D / sqrt(J) <= h). Precision targets remain valid even when no null hypothesis is rejected!
    - Disjoint Confirmatory Seed Block & Pilot Discarding: Pilot rows are DISCARDED from final inference. This guarantees independence between planned count J and final confirmatory data, preserving exact nominal Type I error (alpha) under normal contrast models via iterated expectation.
    - Confirmatory Execution: Solves J_final = min(J_max, max_k max(J_prec,k, J_pow,k)). Generates J_final independent replications on a disjoint seed block and evaluates final inference ONCE at adjusted level alpha_adj.

  * Layer 2: Decision Layer (Statistical Power & Multiplicity)
    - Common Random Numbers & Paired Contrasts: Formed on common replications (D_{k,j} = L_{A,j} - L_{B,j}). Induces positive correlation rho between models, reducing sampling variance by a factor of 1 / (1 - rho) and saving substantial compute compared to independent sampling. Listwise deletion across pairs is strictly enforced!
    - Confirmatory Test Modes (M1 to M5):
      * M1: Two-sided equality test (H0: theta_D = 0 vs H1: theta_D != 0)
      * M2: Directional superiority test (H0: theta_D >= 0 vs H1: theta_D < 0, lower loss is better)
      * M3: Minimum-effect superiority test (H0: theta_D >= -Delta vs H1: theta_D < -Delta)
      * M4: Non-inferiority test (H0: theta_D >= Delta vs H1: theta_D < Delta)
      * M5: Equivalence / TOST (H0: |theta_D| >= Delta vs H1: |theta_D| < Delta)
    - Noncentral-t Power Calculation: All power functions use noncentral-t with J-1 degrees of freedom t_{J-1, ncp} with ncp = sqrt(J) * delta / sigma_D (using exact noncentral-t CDF via scipy.stats.nct in Python or pt() in R).
    - Multiplicity Correction Procedures:
      * Bonferroni: FWER control dividing alpha by K (conservative).
      * Holm (Holm-Bonferroni): Step-down FWER control.
      * Benjamini-Hochberg (BH): FDR control (useful for exploratory screening across many benchmarks).
      * Romano-Wolf Stepdown: Bootstrap-based FWER control that accounts for correlation among paired contrasts.
    - Family Success Criteria:
      * Conjunctive Power (pi_cap): Probability that ALL K primary contrast claims hold simultaneously.
      * Disjunctive Power (pi_cup): Probability that AT LEAST ONE primary contrast claim holds.
    - Model Confidence Set (MCS) Layer (Hansen, Lunde & Nason 2011): Post-hoc screening procedure using White's Reality Check or Hansen's SPA (T_R and T_max statistics) to output the non-dominated model set M*_{1-alpha}.

  * Optional Adaptive Mode (Algorithm 2 - Internal Pilot Re-estimation):
    - Reuses pilot rows across adaptive looks (batch size B, max looks l_max, budget cap J_max).
    - WARNING: Reusing pilot rows inflates unconditional Type I error; adaptive runs must flag that error rates are non-nominal and measured.

==============================================================================
3. PERFORMANCE ESTIMANDS TAXONOMY
==============================================================================
- PEHE (Precision in Estimating Heterogeneous Effects): Root mean of replication-level mean squared errors in CATE estimates.
- CATE RMSE: Averages unrooted MSE across replications and takes square root only after across-replication averaging. Note: E(sqrt(Q_j)) != sqrt(E(Q_j)).
- ATE Error: Replication-level error (tau_hat_j - tau_j)^2.
- Coverage & Sharpness: Coverage is center-calibrated around nominal level 1 - c (|E(C_j) - (1-c)| <= delta). Over-coverage is uninformative and penalized. Sharpness (mean interval width) is reported separately.
- Interval Score (IS_{c,j}): Strictly proper scoring rule for central 1 - c intervals:
    IS_{c,j} = (1/n) sum_{i=1}^n [ (u_{j,i} - l_{j,i}) + (2/c)(l_{j,i} - x_{j,i}) 1_{x_{j,i} < l_{j,i}} + (2/c)(x_{j,i} - u_{j,i}) 1_{x_{j,i} > u_{j,i}} ]
  Lower IS values indicate superior predictive accuracy and sharp uncertainty calibration.

==============================================================================
4. CODE IMPLEMENTATION GUIDELINES
==============================================================================

PYTHON MODULE-LEVEL USAGE (tisca):
```python
import numpy as np
from tisca import inference, planning

# 1. Independent Pilot (J0 replications)
pilot_A = np.asarray(pilot_results["method_A"], dtype=float)
pilot_B = np.asarray(pilot_results["method_B"], dtype=float)
pilot_D = pilot_A - pilot_B
J0 = pilot_D.size

# 2. Plan J_final with Chi-square variance assurance & targets
J_final, sigma_ub = planning.required_J(
    np.std(pilot_D, ddof=1),
    J0,
    gamma=0.20,             # 80% assurance buffer
    mode="M2",              # lower-is-better directional superiority
    delta=-0.20,            # planned advantage for method A
    target_mcse=0.05,       # precision target
    target_power=0.80,      # power target
    alpha=0.05,             # significance level (use alpha/K for Bonferroni)
    J_max=10000,
)
print({"J_final": J_final, "sigma_upper_bound": sigma_ub})

# 3. Confirmatory Phase (J_final replications on disjoint seed block with CRN)
confirm_A = np.asarray(confirm_results["method_A"], dtype=float)
confirm_B = np.asarray(confirm_results["method_B"], dtype=float)
confirm_D = confirm_A - confirm_B

final = inference.paired_t(confirm_D, alternative="less")
print({"estimate": final["estimate"], "p_value": final["p_value"]})
```

PYTHON PROCEDURAL RUNNER USAGE (tisca.procedure):
```python
from tisca.procedure import TwoStageDesign

def sim_func(seed):
    # Execute one replication for seed, return row vector of metrics
    return metrics_array # shape (1, n_metrics)

primary_contrasts = [
    {
        "name": "ModelA vs ModelB (PEHE)",
        "A": 0, "B": 1,
        "mode": "M1",
        "delta": -0.05,
        "target_type": "power",
        "target_power": 0.80,
    }
]

design = TwoStageDesign(
    sim_func=sim_func,
    primary_contrasts=primary_contrasts,
    J0=50,
    gamma=0.20,
    alpha=0.05,
    correction="bonferroni",
    J_max=1000,
    n_metrics=2
)
results = design.run(verbose=True)
```

R MODULE-LEVEL USAGE (tisca):
```r
library(tisca)

# 1. Independent Pilot (J0 replications)
pilot_A <- pilot_results$method_A
pilot_B <- pilot_results$method_B
pilot_D <- pilot_A - pilot_B
J0 <- length(pilot_D)

# 2. Plan J_final with Chi-square variance assurance & targets
sigma_upper <- sigma_ub(sd(pilot_D), J0 = J0, gamma = 0.20)
power_plan <- solve_power_J(
  mode = "M2", delta = -0.20, sigma_D = sigma_upper,
  alpha_adj = 0.05, target_power = 0.80, J_max = 10000
)
mcse_plan <- solve_mcse_J(sigma_upper, m = 0.05, J_max = 10000)
J_final <- combine_J(c(power_plan$J, mcse_plan$J), J_max = 10000)$J_final
print(list(J_final = J_final, sigma_upper_bound = sigma_upper))

# 3. Confirmatory Phase (J_final replications on disjoint seeds with CRN)
confirm_A <- confirm_results$method_A
confirm_B <- confirm_results$method_B
contrast <- contrast_from_columns(confirm_A, confirm_B)
final <- paired_t(contrast$D, alternative = "less")
print(list(estimate = final$estimate, p_value = final$p_value))
```

R PROCEDURAL RUNNER USAGE (tisca):
```r
library(tisca)

sim_func <- function(seed) {
  set.seed(seed)
  data.frame(model_a_pehe = rgamma(1, 2, 4), model_b_pehe = rgamma(1, 2.5, 4))
}

primary <- list(
  list(
    metric_a = "model_a_pehe",
    metric_b = "model_b_pehe",
    mode = "M1",
    delta = -0.05,
    target_type = "power",
    target_power = 0.80,
    J0 = 50,
    gamma = 0.20
  )
)

res <- run_two_stage(
  sim_func = sim_func,
  primary = primary,
  J0 = 50,
  alpha = 0.05,
  correction = "bonferroni",
  J_max = 1000,
  verbose = TRUE
)
```

==============================================================================
5. INSTRUCTIONS FOR RESPONSE GENERATION
==============================================================================
- Introduce yourself as the TISCA Helper Bot (v2).
- Be precise, rigorous, and supportive. Use LaTeX for mathematical formulas when helpful.
- Provide clean Python or R snippets matching TISCA v2 syntax (`from tisca import inference, planning` or `tisca.procedure.TwoStageDesign` in Python, `library(tisca)` in R).
- Always emphasize the importance of listwise paired contrasts, CRN (common random numbers), pilot row discarding for independent confirmatory inference, and precision/power targets.
"""

# Initialize Gemini / Gemma model with safety fallback
@st.cache_resource
def get_model(model_name: str):
    try:
        m = genai.GenerativeModel(
            model_name,
            system_instruction=TISCA_SYSTEM_PROMPT,
            generation_config=genai.types.GenerationConfig(
                temperature=0.7,
                top_p=0.95,
            )
        )
        return m, model_name
    except Exception as err:
        fallback_name = "gemini-2.5-flash"
        m = genai.GenerativeModel(
            fallback_name,
            system_instruction=TISCA_SYSTEM_PROMPT,
            generation_config=genai.types.GenerationConfig(
                temperature=0.7,
                top_p=0.95,
            )
        )
        return m, f"{model_name} (Fallback to {fallback_name}: {err})"

model, active_model_info = get_model(MODEL_NAME)

# Initialize chat session in Streamlit session_state
if "chat_session" not in st.session_state:
    st.session_state.chat_session = model.start_chat(history=[])

# --- Sidebar UI ---
with st.sidebar:
    st.image("https://img.shields.io/badge/TISCA-v2.0.0--dev-blue?style=for-the-badge", use_container_width=True)
    st.title("⚙️ TISCA v2 Controls")
    
    st.info(f"**LLM Model:** {MODEL_NAME}")
    
    st.markdown("### 🚀 Installation")
    st.markdown("**Python:**")
    st.code('pip install "git+https://github.com/hugogobato/Test-Informed-Simulation-Count-Algorithm-TISCA.git"', language="bash")
    st.markdown("**R:**")
    st.code('remotes::install_github("hugogobato/Test-Informed-Simulation-Count-Algorithm-TISCA", subdir = "tisca")', language="r")
    
    st.markdown("---")
    st.markdown("### 📚 Quick Reference")
    st.markdown("""
    - **Protocol:** Two-Stage Default (Algorithm 1)
    - **Pilot Assurance:** 1-sided upper $\\chi^2$ bound (\\gamma = 0.20, 80% assurance)
    - **Precision Target:** MCSE $m$ or CI half-width $h$
    - **Decision Layer:** Power target $\\pi(J; \\delta, \\sigma_D, \\alpha)$
    - **Multiplicity:** Bonferroni, Holm, BH, Romano-Wolf
    - **Test Modes:** M1 (2-sided), M2 (Superiority), M3 (Min-effect), M4 (Non-inf), M5 (Equivalence)
    """)
    
    st.markdown("---")
    st.markdown("### 📄 Paper Reference")
    st.caption("Gobato Souto & Louzada (2026)\n*Beyond Arbitrary Replications: A Principled Approach to Simulation Design in Causal Inference*")
    
    if st.button("🔄 Clear Chat History"):
        st.session_state.chat_session = model.start_chat(history=[])
        st.rerun()

# --- Main UI Header ---
st.title("🧪 TISCA Helper Bot (v2)")
st.caption("Powered by **Gemma 4 31B** | Test-Informed Simulation Count Algorithm")

st.markdown("""
Welcome! I am your AI pair-programmer and methodological guide for **TISCA v2**.
Whether you are designing a new Monte Carlo evaluation of treatment-effect estimators or debugging simulation code in **Python** or **R**, I am here to help.
""")

# Quick Reference Tabs
tab_install, tab_overview, tab_modes, tab_python, tab_r = st.tabs([
    "💻 Installation & Setup",
    "📊 Overview & Protocol",
    "🎯 Test Modes & Estimands",
    "🐍 Python Examples",
    "🔵 R Examples"
])

with tab_install:
    st.markdown("""
    ### Installation Instructions

    TISCA v2 is distributed directly from the [TISCA GitHub repository](https://github.com/hugogobato/Test-Informed-Simulation-Count-Algorithm-TISCA).

    #### Python (>= 3.9)
    Install directly from GitHub:
    ```bash
    python -m pip install "git+https://github.com/hugogobato/Test-Informed-Simulation-Count-Algorithm-TISCA.git"
    ```
    Then import the v2 modules:
    ```python
    from tisca import inference, planning
    ```

    #### R (>= 4.0)
    Install from the `tisca/` subdirectory using `remotes`:
    ```r
    install.packages("remotes")
    remotes::install_github(
      "hugogobato/Test-Informed-Simulation-Count-Algorithm-TISCA",
      subdir = "tisca"
    )
    library(tisca)
    ```
    """)

with tab_overview:
    st.markdown("""
    #### The Two-Stage Default Protocol (Algorithm 1)
    1. **Declare Replication Unit & Estimands:** Define $L_j$ (e.g. PEHE, ATE error, Interval Score) and primary paired contrasts $D_{k,j} = L_{A,j} - L_{B,j}$ using Common Random Numbers (CRN).
    2. **Independent Pilot ($J_0$):** Run $J_0$ replications to estimate sample contrast standard deviation $s_D$.
    3. **Variance Assurance Buffer:** Calculate $\\hat{\\sigma}_{D,\\text{UB}} = s_D \\sqrt{\\frac{J_0 - 1}{\\chi^2_{\\gamma, J_0 - 1}}}$ (default $\\gamma = 0.20$, 80% assurance).
    4. **Solve Replication Count ($J_{\\text{final}}$):** Compute required counts $J_{\\text{prec}}$ and $J_{\\text{pow}}$ across active precision & power targets. Set $J_{\\text{final}} = \\min(J_{\\text{max}}, \\max_k(J_k))$.
    5. **Discard Pilot & Run Confirmatory Block:** Discard pilot rows from final inference to preserve exact Type I error $(\\alpha)$. Run $J_{\\text{final}}$ independent replications on a disjoint seed block and perform the final test once at $\\alpha_{\\adj}$.
    """)

with tab_modes:
    st.markdown("""
    #### Hypothesis Modes (M1–M5)
    | Mode | Description | Null Hypothesis ($H_0$) | Planning Noncentrality |
    | :--- | :--- | :--- | :--- |
    | **M1** | Two-sided equality | $\\theta_D = 0$ | $\\lambda = \\sqrt{J}\\delta / \\sigma_D$ |
    | **M2** | Directional superiority | $\\theta_D \\ge 0$ (lower loss is better) | $\\lambda = \\sqrt{J}\\delta / \\sigma_D$ |
    | **M3** | Minimum-effect superiority | $\\theta_D \\ge -\\Delta$ | $\\lambda = \\sqrt{J}(\\delta + \\Delta) / \\sigma_D$ |
    | **M4** | Non-inferiority | $\\theta_D \\ge \\Delta$ | $\\lambda = \\sqrt{J}(\\delta - \\Delta) / \\sigma_D$ |
    | **M5** | Equivalence (TOST) | $|\\theta_D| \\ge \\Delta$ | Dual noncentral-$t$ |
    """)

with tab_python:
    st.markdown("#### Basic Workflow (Direct Module Access)")
    st.code("""
import numpy as np
from tisca import inference, planning

# 1. Independent Pilot (J0 replications)
pilot_A = np.asarray(pilot_results["method_A"], dtype=float)
pilot_B = np.asarray(pilot_results["method_B"], dtype=float)
pilot_D = pilot_A - pilot_B

J0 = pilot_D.size
J_final, sigma_ub = planning.required_J(
    np.std(pilot_D, ddof=1),
    J0,
    gamma=0.20,
    mode="M2",             # lower-is-better directional superiority
    delta=-0.20,            # planned advantage for method A
    target_mcse=0.05,
    target_power=0.80,
    alpha=0.05,             # use 0.05 / K for K Bonferroni-planned contrasts
    J_max=10000,
)
print({"J_final": J_final, "sigma_upper_bound": sigma_ub})

# 2. Confirmatory Phase (J_final replications on disjoint seed block with CRN)
confirm_A = np.asarray(confirm_results["method_A"], dtype=float)
confirm_B = np.asarray(confirm_results["method_B"], dtype=float)
confirm_D = confirm_A - confirm_B

final = inference.paired_t(confirm_D, alternative="less")
print({"estimate": final["estimate"], "p_value": final["p_value"]})
""", language="python")

    st.markdown("#### Procedural Runner Workflow (`TwoStageDesign`)")
    st.code("""
from tisca.procedure import TwoStageDesign

def sim_func(seed):
    # Execute replication for seed, return row vector of metrics
    return metrics_array # shape (1, n_metrics)

design = TwoStageDesign(
    sim_func=sim_func,
    primary_contrasts=[{
        "name": "ModelA vs ModelB",
        "A": 0, "B": 1,
        "mode": "M2",
        "delta": -0.20,
        "target_type": "power",
        "target_power": 0.80,
    }],
    J0=50,
    gamma=0.20,
    alpha=0.05,
    J_max=10000,
    n_metrics=2
)
results = design.run(verbose=True)
""", language="python")

with tab_r:
    st.markdown("#### Basic Workflow (Direct Module Access)")
    st.code("""
library(tisca)

# 1. Independent Pilot (J0 replications)
pilot_A <- pilot_results$method_A
pilot_B <- pilot_results$method_B
pilot_D <- pilot_A - pilot_B

J0 <- length(pilot_D)
sigma_upper <- sigma_ub(sd(pilot_D), J0 = J0, gamma = 0.20)
power_plan <- solve_power_J(
  mode = "M2", delta = -0.20, sigma_D = sigma_upper,
  alpha_adj = 0.05, target_power = 0.80, J_max = 10000
)
mcse_plan <- solve_mcse_J(sigma_upper, m = 0.05, J_max = 10000)
J_final <- combine_J(c(power_plan$J, mcse_plan$J), J_max = 10000)$J_final
print(list(J_final = J_final, sigma_upper_bound = sigma_upper))

# 2. Confirmatory Phase (J_final replications on disjoint seeds with CRN)
confirm_A <- confirm_results$method_A
confirm_B <- confirm_results$method_B
contrast <- contrast_from_columns(confirm_A, confirm_B)
final <- paired_t(contrast$D, alternative = "less")
print(list(estimate = final$estimate, p_value = final$p_value))
""", language="r")

    st.markdown("#### Procedural Runner Workflow (`run_two_stage`)")
    st.code("""
library(tisca)

sim_func <- function(seed) {
  set.seed(seed)
  data.frame(
    model_a_pehe = rgamma(1, shape = 2.0, scale = 0.5),
    model_b_pehe = rgamma(1, shape = 2.5, scale = 0.5)
  )
}

primary <- list(
  list(
    metric_a = "model_a_pehe",
    metric_b = "model_b_pehe",
    mode = "M2",
    delta = -0.20,
    target_type = "power",
    target_power = 0.80,
    J0 = 50,
    gamma = 0.20
  )
)

results <- run_two_stage(
  sim_func = sim_func,
  primary = primary,
  J0 = 50,
  alpha = 0.05,
  correction = "bonferroni",
  J_max = 10000,
  verbose = TRUE
)
""", language="r")

st.markdown("---")

# Display Chat History
for message in st.session_state.chat_session.history:
    role = "user" if message.role == "user" else "assistant"
    with st.chat_message(role):
        if role == "assistant":
            st.markdown(message.parts[0].text)
        else:
            st.markdown(message.parts[0].text)

# User Prompt Input
user_prompt = st.chat_input("Ask TISCA Helper Bot about TISCA v2, installation, or paste R/Python code...")

if user_prompt:
    st.chat_message("user").markdown(user_prompt)
    try:
        with st.spinner("TISCA Helper Bot (Gemma 4 31B) is thinking..."):
            response = st.session_state.chat_session.send_message(user_prompt)
        with st.chat_message("assistant"):
            st.markdown(response.text)
    except Exception as e:
        st.error(f"An error occurred while communicating with Gemma 4 31B: {e}")