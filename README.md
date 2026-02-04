# Text-to-SQL

A Streamlit app that converts natural language questions into SQL and runs them against a MySQL database. Built with LangChain, Google Gemini, and RAGAS-based evaluation.

**Live app:** [https://text-to-sql-986401037066.asia-south1.run.app/](https://text-to-sql-986401037066.asia-south1.run.app/)

**Demo video:** [https://drive.google.com/file/d/12eEhAIS7-PaUDshuGGoK6zx0v_W8PomK/view](https://drive.google.com/file/d/12eEhAIS7-PaUDshuGGoK6zx0v_W8PomK/view)


---

## Features

- **Natural language → SQL:** Ask questions in plain English; the app uses Gemini to generate SQL and executes it against the database.
- **Schema browser:** View the database schema (tables and columns) in the UI.
- **Query history:** See recent questions and the generated SQL.
- **Evaluation metrics:** RAGAS-based evaluation (context precision, helpfulness) with predefined questions. Results show ground-truth SQL and model-generated SQL side by side in expanders.
- **Table-name normalization:** Generated SQL is normalized so table names match the database (e.g. `Customers` → `customers`) for case-sensitive MySQL (e.g. Cloud SQL).

---

## Tech stack

- **UI:** Streamlit  
- **SQL generation:** LangChain, Google Gemini (ChatGoogleGenerativeAI)  
- **Evaluation:** RAGAS, Groq, HuggingFace embeddings  
- **Database:** MySQL (local or Google Cloud SQL)  
- **Deployment:** Docker, Google Cloud Run, Artifact Registry, GitHub Actions  

---

## Project structure

```
.
├── app.py                 # Streamlit app entry
├── main.py                # CLI entry (runs example script)
├── requirements.txt
├── Dockerfile
├── text_to_sql.sql        # DB dump for import
├── text_to_sql/
│   ├── chains.py          # SQL-generation chain (schema + prompt + LLM)
│   ├── config.py          # DB and LLM config from env
│   ├── db.py              # MySQL connection, schema, run_query, table-name normalizer
│   ├── evaluation.py      # RAGAS evaluation (dataset, metrics)
│   ├── llm_models.py      # Gemini (query), Groq + HF (evaluation)
│   └── prompts.py         # SQL prompt template
└── scripts/
    ├── run_example.py     # Single-question example
    └── run_ragas_eval.py  # Batch RAGAS evaluation
```

---

## Local setup

1. **Python:** 3.12 recommended.

2. **Environment variables** (e.g. in `.env`):

   - `DB_HOST`, `DB_PORT`, `DB_USER`, `DB_PASSWORD`, `DB_NAME` — MySQL connection.
   - `GOOGLE_API_KEY`, `GOOGLE_MODEL_NAME` — Gemini (SQL generation).
   - `GROQ_API_KEY`, `GROQ_MODEL_NAME` — Groq (RAGAS evaluation).

3. **Install and run:**

   ```bash
   pip install -r requirements.txt
   streamlit run app.py
   ```

4. **Database:** Use an existing MySQL instance or import `text_to_sql.sql` (e.g. `mysql -u user -p text_to_sql < text_to_sql.sql` or Cloud SQL Import from a GCS bucket).

---

## Deployment

- **Container:** Built with the repo `Dockerfile` (Python 3.12 slim).
- **CI/CD:** GitHub Actions (`.github/workflows/deploy.yml`) builds the image, pushes to Google Artifact Registry, and deploys to **Cloud Run** on push to `main`.
- **Database:** Cloud SQL for MySQL; data can be migrated via `mysqldump` → upload to Cloud Storage → Cloud SQL Import.
- **Secrets:** GitHub secrets for `GCP_SA_KEY`, `GCP_PROJECT_ID`, DB and API keys; GCP service account needs Artifact Registry Writer, Cloud Run Admin, Service Account User.

---

## Evaluation

- **Ground truth:** A fixed set of natural-language questions and reference SQL queries (hardcoded in `app.py` and `scripts/run_ragas_eval.py`).
- **Metrics:** Context Precision, Rubrics Score (helpfulness). Results (including ground-truth and model SQL) are shown in the app and can be cached.

---
