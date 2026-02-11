# Text-to-SQL

A Streamlit app that converts natural language questions into SQL and runs them against a MySQL database. Built with LangChain, Google Gemini, and RAGAS-based evaluation.

**Live app: New link** [https://text-to-sql-343287988900.asia-south1.run.app/](https://text-to-sql-343287988900.asia-south1.run.app/)

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

### Step-by-step: Local test against Cloud SQL (before deploy)

Use the **Cloud SQL Auth Proxy** so the app talks to your real Cloud SQL instance from your machine. This verifies database, schema, and credentials before you deploy.

---

**Step 1 — Install Google Cloud SDK (if needed)**  
- If you don’t have `gcloud` and `cloud-sql-proxy`:
  - Install [Google Cloud SDK](https://cloud.google.com/sdk/docs/install) for Windows.
  - Open a new terminal and run: `gcloud init` (login and select your project).

**Step 2 — Install Cloud SQL Auth Proxy**  
- **Option A (recommended on Windows):** Download the proxy:
  - Go to: https://cloud.google.com/sql/docs/mysql/connect-auth-proxy#install  
  - Under “Download the proxy”, choose **Windows 64-bit**, download the `.exe`.  
  - Rename it to `cloud-sql-proxy.exe` and put it in a folder that’s on your PATH (e.g. `C:\Program Files\Google\Cloud SQL Auth Proxy\`) or in your project folder.  
- **Option B:** If you use gcloud components:
  ```powershell
  gcloud components install cloud-sql-proxy
  ```
- Check it works:
  ```powershell
  cloud-sql-proxy --version
  ```

**Step 3 — Log in to GCP (application default credentials)**  
- In PowerShell or Command Prompt:
  ```powershell
  gcloud auth application-default login
  ```
- A browser window opens; sign in with the Google account that has access to your Cloud SQL project.  
- When it says “Credentials saved”, you’re done.

**Step 4 — Start the proxy (leave this terminal open)**  
- Replace `YOUR_PROJECT_ID` with your actual GCP project ID (e.g. `text-to-sql-486913`). Use the same instance name and region as in deploy (e.g. `text-to-sql-db`, `asia-south1`).  
- Use port `3307` so it doesn’t clash with a local MySQL on 3306:
  ```powershell
  cloud-sql-proxy "YOUR_PROJECT_ID:asia-south1:text-to-sql-db" --port=3307
  ```
- You should see something like: `Listening on 127.0.0.1:3307 for YOUR_PROJECT_ID:asia-south1:text-to-sql-db`.  
- Leave this terminal running for the whole test.

**Step 5 — Configure the app to use the proxy**  
- In your project folder, create or edit `.env` (same directory as `app.py`). Set:
  ```env
  DB_HOST=127.0.0.1
  DB_PORT=3307
  DB_USER=your_cloud_sql_username
  DB_PASSWORD=your_cloud_sql_password
  DB_NAME=text_to_sql
  GOOGLE_API_KEY=your_gemini_api_key
  GROQ_API_KEY=your_groq_api_key
  ```
- Use the same `DB_USER`, `DB_PASSWORD`, and `DB_NAME` as in your Cloud SQL instance (and GitHub secrets).  
- Ensure the database `text_to_sql` (or whatever you set as `DB_NAME`) exists on Cloud SQL and has the schema (e.g. import `text_to_sql.sql` if needed).

**Step 6 — Install Python deps and run the app**  
- In a **new** terminal (the first one is still running the proxy), go to the project folder:
  ```powershell
  cd "c:\Pala Ravikanth\GenAI Projects\Text to SQL"
  pip install -r requirements.txt
  streamlit run app.py
  ```
- When Streamlit opens in the browser, try a natural-language question.  
- If it runs and returns results, your local test against Cloud SQL is successful.

**Step 7 — Stop when done**  
- In the Streamlit terminal: `Ctrl+C`.  
- In the proxy terminal: `Ctrl+C`.

---

This flow uses the normal TCP connection (host:port), not the `/cloudsql/` socket used on Cloud Run, but it confirms the database and app work before you deploy.

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
