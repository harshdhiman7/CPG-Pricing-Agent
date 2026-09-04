# 🏷️ Autonomous Retail Pricing Agent

An agentic retail pricing and econometric analytics application powered by local LLMs (**Qwen 2.5 via Ollama**), **Streamlit**, and **Scikit-Learn**.

This system processes weekly point-of-sale (POS) dataset formats (such as Circana or NielsenIQ scanner data) to isolate baseline **Price Elasticity of Demand (PED)** from promotional noise using **Multivariate Log-Log OLS Regression**, delivering executive-ready pricing strategies with zero data leakage.

---

## 🌟 Key Features

- **🤖 Agentic Tool Orchestration** — Uses native LLM function-calling to autonomously decide when and how to execute quantitative analytical tools.
- **📈 Log-Log OLS Econometrics** — Replaces naive elasticity calculations with a multivariate linear regression model ($\ln(\text{UNITS}) \sim \ln(\text{PRICE}) + \text{Promotions}$) to isolate ceteris paribus price sensitivity.
- **📊 Promotional Lift Isolation** — Controls for confounding trade marketing variables including Features, Displays, and Temporary Price Reductions (TPR).
- **🔒 Privacy-First & Local** — Runs entirely on your local machine using Ollama, ensuring proprietary POS data never leaves your environment.
- **💻 Interactive UI** — Built with Streamlit for seamless CSV uploads, interactive query execution, and rapid data exploration.

---

## 🛠️ Tech Stack & Architecture

- **LLM Orchestration**: Qwen 2.5 (`qwen2.5:3b` or `qwen2.5:7b`) via [Ollama](https://ollama.com/)
- **Frontend / App Framework**: Streamlit
- **Data Processing & Econometrics**: Pandas, NumPy, Scikit-Learn (OLS Linear Regression)

```

                 ┌─────────────────────────┐
                 │  User Input (Streamlit) │
                 └────────────┬────────────┘
                              │
                              ▼
                 ┌─────────────────────────┐
                 │     LLM Orchestrator    │
                 │       (Qwen 2.5)        │
                 └────────────┬────────────┘
                              │
                    Determines Tool to Call
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌───────────────┐   ┌──────────────────┐   ┌──────────────────┐
│ Price          │   │ Promo Lift       │   │ Store Segment    │
│ Elasticity     │   │ (Baseline        │   │ (Location &      │
│ (Log-Log OLS)  │   │  Compare)        │   │  Spend)          │
└───────┬────────┘   └────────┬─────────┘   └────────┬─────────┘
        │                     │                       │
        └─────────────────────┼───────────────────────┘
                              │
                  Returns JSON Results to LLM
                              │
                              ▼
                 ┌─────────────────────────┐
                 │     LLM Synthesizer     │
                 │  (Exec Recommendation)  │
                 └─────────────────────────┘

```

---

## 📋 Prerequisites

1. **Python 3.9+** installed on your system.
2. **Ollama** installed and running locally. Download from [ollama.com](https://ollama.com/).

---

## 🚀 Quickstart Guide

### 1. Pull the Lightweight Qwen Model

Open your terminal and pull the Qwen 2.5 model:

```bash
ollama pull qwen2.5:3b
```

*(Optional: you can also use `ollama pull qwen2.5:7b` for higher reasoning accuracy.)*

### 2. Install Required Python Packages

```bash
pip install streamlit pandas numpy scikit-learn ollama
```

### 3. Launch the Streamlit App

Navigate to the directory containing `pricing_agent.py` and run:

```bash
streamlit run pricing_agent.py
```

---

## 📊 Expected CSV Dataset Schema

The application accepts standard weekly POS / scanner datasets containing the following columns:

| Column Name      | Type          | Description                                    |
|-------------------|---------------|-------------------------------------------------|
| `WEEK_END_DATE`   | Date          | Reporting week end date (`YYYY-MM-DD`)          |
| `STORE_ID`        | Integer       | Unique identifier for store location            |
| `UPC`             | Integer       | Universal Product Code (SKU identifier)         |
| `UNITS`           | Numeric       | Volume units sold                               |
| `PRICE`           | Numeric       | Actual shelf price charged                      |
| `BASE_PRICE`      | Numeric       | Regular non-promotional price                   |
| `FEATURE`         | Binary (0/1)  | Circular or ad feature flag                     |
| `DISPLAY`         | Binary (0/1)  | In-store display promo flag                     |
| `TPR_ONLY`        | Binary (0/1)  | Temporary Price Reduction flag                  |
| `DESCRIPTION`     | Text          | Product description                             |
| `SPEND`           | Numeric       | Total dollar sales revenue                      |
| `SEG_VALUE_NAME`  | Text          | Store tier/segment (e.g., Upscale, Value)       |

---

## 💡 Example Queries to Ask

Once your CSV is uploaded into the app, try asking:

- *"Analyze UPC 123456. What is its multivariate price elasticity, promo impact, and recommended price strategy?"*
- *"What is the promotional lift for UPC 987654 when placed on Display vs. Feature?"*
- *"Show store segment metrics for Store ID 101 and summarize overall spend performance."*

---

## ⚙️ How the Econometric Model Works

Instead of calculating simple percentage changes (which introduces noise from promotions), the agent runs a **Log-Log Ordinary Least Squares (OLS)** regression:

$$\ln(\text{UNITS}) = \beta_0 + \beta_1 \ln(\text{PRICE}) + \beta_2 \text{FEATURE} + \beta_3 \text{DISPLAY} + \beta_4 \text{TPR\_ONLY} + \epsilon$$

- **$\beta_1$ (Price Elasticity)** — Represents the true ceteris paribus elasticity of demand.
- **$\beta_2, \beta_3, \beta_4$** — Isolate percentage volume uplifts directly attributable to trade marketing activities.

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.