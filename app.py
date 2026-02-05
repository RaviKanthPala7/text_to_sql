"""
Streamlit UI for Text-to-SQL Query Interface.

This application provides a web interface for:
- Querying the database using natural language
- Viewing database schema
- Running evaluation metrics
- Viewing query history
"""

import json
import logging
import os
import warnings
from pathlib import Path
from typing import Any

# Reduce noisy torch/datasets logs from HuggingFace (e.g. "Examining the path of torch.classes")
logging.getLogger("torch").setLevel(logging.ERROR)
logging.getLogger("datasets").setLevel(logging.ERROR)

import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

from text_to_sql.chains import (
    batch_generate_sql_queries,
    create_intent_validation_chain,
    create_sql_chain,
    generate_sql_query,
    validate_query_intent,
)
from text_to_sql.config import get_llm_config
from text_to_sql.db import get_database, get_schema, normalize_sql_table_names, run_query
from text_to_sql.evaluation import (
    build_evaluation_components,
    build_evaluation_dataset,
    run_ragas_evaluation,
)
from text_to_sql.llm_models import get_query_llm

# Cache file for evaluation results
EVALUATION_CACHE_FILE = Path("evaluation_cache.json")

# Page configuration
st.set_page_config(
    page_title="Text-to-SQL Query Interface",
    page_icon="🗄️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize session state for caching and history
if "db" not in st.session_state:
    st.session_state.db = None
if "llm" not in st.session_state:
    st.session_state.llm = None
if "sql_chain" not in st.session_state:
    st.session_state.sql_chain = None
if "intent_chain" not in st.session_state:
    st.session_state.intent_chain = None
if "query_history" not in st.session_state:
    st.session_state.query_history = []
if "schema_displayed" not in st.session_state:
    st.session_state.schema_displayed = False


def initialize_components():
    """Initialize database, LLM, and chain (cached in session state)."""
    if st.session_state.db is None:
        try:
            with st.spinner("Connecting to database..."):
                st.session_state.db = get_database()
        except Exception as e:
            st.error(f"Failed to connect to database: {str(e)}")
            st.info("Please check your database configuration in the .env file.")
            raise

    if st.session_state.llm is None:
        try:
            with st.spinner("Initializing LLM..."):
                st.session_state.llm = get_query_llm()
        except Exception as e:
            st.error(f"Failed to initialize LLM: {str(e)}")
            st.info("Please check your GOOGLE_API_KEY in the .env file.")
            raise

    if st.session_state.intent_chain is None:
        try:
            st.session_state.intent_chain = create_intent_validation_chain(
                st.session_state.llm
            )
        except Exception as e:
            st.error(f"Failed to create intent validation chain: {str(e)}")
            raise
    if st.session_state.sql_chain is None:
        try:
            st.session_state.sql_chain = create_sql_chain(
                st.session_state.db, st.session_state.llm
            )
        except Exception as e:
            st.error(f"Failed to create SQL chain: {str(e)}")
            raise


def load_evaluation_cache() -> dict | None:
    """Load cached evaluation results if they exist."""
    if EVALUATION_CACHE_FILE.exists():
        try:
            with open(EVALUATION_CACHE_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def save_evaluation_cache(data: dict) -> None:
    """Save evaluation results to cache file."""
    try:
        with open(EVALUATION_CACHE_FILE, "w") as f:
            json.dump(data, f, indent=2, default=str)
    except Exception as e:
        st.warning(f"Failed to save cache: {str(e)}")


def format_evaluation_result(result) -> pd.DataFrame:
    """Convert RAGAS evaluation result to aggregate metrics DataFrame."""
    # Get the full results first
    if hasattr(result, "to_pandas"):
        df = result.to_pandas()
    elif isinstance(result, dict):
        if "metrics" in result:
            df = pd.DataFrame(result["metrics"])
        else:
            df = pd.DataFrame([result])
    elif isinstance(result, pd.DataFrame):
        df = result
    else:
        try:
            result_dict = json.loads(str(result))
            df = pd.DataFrame([result_dict])
        except Exception:
            df = pd.DataFrame({"Result": [str(result)]})
    
    # Calculate aggregate metrics (mean/average) for numeric columns
    if isinstance(df, pd.DataFrame) and len(df) > 0:
        # Filter out non-metric columns (like 'reference', 'user_input', etc.)
        # Keep only metric columns (context_precision, helpfulness, etc.)
        metric_columns = [col for col in df.columns if col not in ['reference', 'user_input', 'response', 'retrieved_contexts']]
        
        # Get numeric metric columns
        numeric_cols = [col for col in metric_columns if df[col].dtype in [float, int]]
        
        if len(numeric_cols) > 0:
            # Calculate mean for each metric column
            aggregate_data = {}
            for col in numeric_cols:
                aggregate_data[col] = round(df[col].mean(), 4)  # Round to 4 decimal places
            
            # Return as single-row DataFrame with only aggregate metrics
            return pd.DataFrame([aggregate_data])
        else:
            # If no numeric columns, return first row or summary
            return df.iloc[[0]] if len(df) > 0 else df
    
    return df


def format_query_result(result: Any) -> pd.DataFrame:
    """
    Format query result for clean table display like SQL Workbench.
    
    Now that run_query() returns DataFrames directly, this mainly
    handles edge cases and ensures we always have a DataFrame.
    """
    # If already a DataFrame, return as-is
    if isinstance(result, pd.DataFrame):
        return result
    
    # If it's a list of dictionaries, convert directly
    if isinstance(result, list) and result and isinstance(result[0], dict):
        return pd.DataFrame(result)
    
    # If it's a list of tuples/lists, convert to DataFrame
    if isinstance(result, (list, tuple)):
        if result and isinstance(result[0], (list, tuple)):
            return pd.DataFrame(result)
        else:
            return pd.DataFrame({"Result": result})
    
    # Fallback: convert to string and display
    return pd.DataFrame({"Result": [str(result)]})


# Main UI
st.title("🗄️ Text-to-SQL Query Interface")
st.markdown("Ask questions about your database in natural language!")

# Note: Components are initialized lazily when needed (not on page load)

# Sidebar for schema and history
with st.sidebar:
    st.header("📊 Database Information")
    
    if st.button("📋 Show Database Schema", use_container_width=True):
        try:
            # Initialize DB if not already done
            if st.session_state.db is None:
                with st.spinner("Connecting to database..."):
                    st.session_state.db = get_database()
            
            schema = get_schema(st.session_state.db)
            st.session_state.schema_displayed = True
            st.session_state.schema_text = schema
        except Exception as e:
            st.error(f"Failed to retrieve schema: {str(e)}")
    
    if st.session_state.schema_displayed and "schema_text" in st.session_state:
        st.subheader("Database Schema")
        st.code(st.session_state.schema_text, language="sql")
    
    st.divider()
    
    st.header("📜 Query History")
    if st.session_state.query_history:
        for idx, entry in enumerate(reversed(st.session_state.query_history[-10:]), 1):
            with st.expander(f"Query #{len(st.session_state.query_history) - idx + 1}: {entry['question'][:50]}..."):
                st.write("**Question:**", entry["question"])
                st.code(entry["sql"], language="sql")
    else:
        st.info("No queries yet. Ask a question to see history here!")

# Model selection
llm_config = get_llm_config()
available_models = [llm_config.google_model]  # Only one model available

st.selectbox(
    "🤖 Select Model for SQL Generation:",
    options=available_models,
    index=0,
    disabled=True,  # Disabled since only one option
    help=f"Currently using {llm_config.google_model} for SQL query generation"
)

# Main content area
col1, col2 = st.columns([3, 1])

with col1:
    question = st.text_area(
        "Ask your question about the database:",
        placeholder="e.g., What is the total 'Line Total' for Geiss Company?",
        height=100,
    )

with col2:
    st.write("")  # Spacing
    st.write("")  # Spacing
    submit_query = st.button("🔍 Submit Query", type="primary", use_container_width=True)

# Query execution section
if submit_query and question.strip():
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        try:
            # Initialize components if not already done
            initialize_components()

            # Guardrail: validate intent before generating SQL
            with st.spinner("Checking query..."):
                allowed, block_reason = validate_query_intent(
                    st.session_state.intent_chain, question
                )
            if not allowed:
                st.error("This request is not allowed.")
                st.info(block_reason)
                st.caption(
                    "Only read-only questions about the database are permitted. "
                    "Deleting, updating, or modifying data is not allowed."
                )
            else:
                with st.spinner("Generating SQL query..."):
                    sql = generate_sql_query(st.session_state.sql_chain, question)
                # Normalize table names to lowercase so queries work on case-sensitive MySQL
                sql = normalize_sql_table_names(sql)

                st.success("SQL query generated successfully!")

                # Display generated SQL
                st.subheader("📝 Generated SQL Query")
                st.code(sql, language="sql")

                # Execute query and display results
                try:
                    with st.spinner("Executing query..."):
                        result = run_query(st.session_state.db, sql)

                    st.subheader("📊 Query Results")
                    formatted_result = format_query_result(result)

                    # Always display as a clean table (like SQL Workbench)
                    st.dataframe(
                        formatted_result,
                        use_container_width=True,
                        hide_index=True,
                    )

                    # Show row count
                    if len(formatted_result) > 0:
                        st.caption(f"📈 {len(formatted_result)} row(s) returned")
                    else:
                        st.info("Query executed successfully but returned no rows.")

                    # Add to history
                    st.session_state.query_history.append({
                        "question": question,
                        "sql": sql,
                        "result": str(result)[:200] + "..." if len(str(result)) > 200 else str(result),
                    })

                except Exception as e:
                    st.error(f"Failed to execute query: {str(e)}")
                    st.info("The SQL query was generated but could not be executed. Check the query syntax or database connection.")

        except Exception as e:
            st.error(f"Failed to generate SQL query: {str(e)}")
            st.info("This might be due to API issues or invalid question format. Please try again.")

elif submit_query:
    st.warning("Please enter a question before submitting.")

# Evaluation metrics section
st.divider()
st.header("📈 Evaluation Metrics")

# Display model information
llm_config = get_llm_config()
st.markdown(
    f"**Model Used:** `{llm_config.google_model}` (SQL Generation) | "
    f"`{llm_config.groq_model}` (Evaluation)\n\n"
    "Evaluation metrics assess the model's performance on a set of predefined questions. "
    "Results are cached to avoid unnecessary API calls."
)

# Check for cached results
cached_results = load_evaluation_cache()
has_cache = cached_results is not None

# Create two columns for buttons
col_refresh, col_view = st.columns([1, 1])

with col_refresh:
    refresh_eval = st.button("🔄 Refresh Evaluation", type="primary", use_container_width=True)

with col_view:
    view_cached = st.button("📊 View Cached Results", type="secondary", use_container_width=True, disabled=not has_cache)

# Only show cached results if user explicitly clicked "View Cached Results"
if view_cached and has_cache and cached_results:
    st.success("📦 Displaying cached evaluation results")
    
    # Show model information if available in cache
    if "model_used" in cached_results or "evaluator_model" in cached_results:
        model_info = f"**Model Used:** `{cached_results.get('model_used', 'N/A')}` (SQL Generation)"
        if "evaluator_model" in cached_results:
            model_info += f" | `{cached_results.get('evaluator_model', 'N/A')}` (Evaluation)"
        st.markdown(model_info)
    
    if "metrics_df" in cached_results and cached_results["metrics_df"]:
        st.subheader("📊 Aggregate Evaluation Metrics")
        metrics_df = pd.DataFrame(cached_results["metrics_df"])
        # Display aggregate metrics (should already be aggregated from cache)
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)
        st.caption("📊 These are aggregate metrics (averages) across all evaluation questions.")
    
    # Show questions with ground truth and model output (if available in cache)
    if "evaluation_questions" in cached_results:
        st.subheader("📋 Questions Used for Evaluation")
        questions = cached_results["evaluation_questions"]
        references = cached_results.get("evaluation_references", [])
        responses = cached_results.get("evaluation_responses", [])
        has_queries = references and responses and len(references) == len(questions) == len(responses)
        for idx, question in enumerate(questions, 1):
            if has_queries:
                with st.expander(f"{idx}. {question}", expanded=(idx == 1)):
                    st.markdown("**Ground truth query**")
                    st.code(references[idx - 1], language="sql")
                    st.markdown("**Model output query**")
                    st.code(responses[idx - 1], language="sql")
            else:
                st.write(f"{idx}. {question}")
    st.info("💡 Click 'Refresh Evaluation' to recalculate metrics with current model.")

# Show message if no cache exists and user hasn't clicked anything
if not has_cache and not refresh_eval and not view_cached:
    st.info("ℹ️ No cached evaluation results found. Click 'Refresh Evaluation' to run evaluation metrics.")

# Run evaluation ONLY if refresh button was explicitly clicked
if refresh_eval:
    try:
        # Initialize components if not already done
        initialize_components()
        
        with st.spinner("Running evaluation metrics (this may take a few minutes)..."):
            # Get schema for context
            schema_context = get_schema(st.session_state.db)
            retrieved_contexts = [schema_context]
            
            # Predefined questions (same as in run_ragas_eval.py)
            user_inputs = [
                "What was the budget of Product 12",
                "What are the names of all products in the products table?",
                "List all customer names from the customers table.",
                "Find the name and state of all regions in the regions table.",
                "What is the name of the customer with Customer Index = 1",
            ]
            
            # Generate SQL queries for all questions
            with st.spinner("Generating SQL queries for evaluation..."):
                responses = batch_generate_sql_queries(
                    st.session_state.sql_chain, user_inputs
                )
            
            # Reference queries
            references = [
                "SELECT `2017 Budgets` FROM `2017_budgets` WHERE `Product Name` = 'Product 12';",
                "SELECT `Product Name` FROM products;",
                "SELECT `Customer Names` FROM customers;",
                "SELECT name, state FROM regions;",
                "SELECT `Customer Names` FROM customers WHERE `Customer Index` = 1;",
            ]
            
            # Build evaluation dataset
            dataset = build_evaluation_dataset(
                user_inputs=user_inputs,
                retrieved_contexts=retrieved_contexts,
                responses=responses,
                references=references,
            )
            
            # Build evaluation components and run evaluation
            with st.spinner("Computing evaluation metrics..."):
                components = build_evaluation_components()
                result = run_ragas_evaluation(dataset, components)
            
            # Convert result to aggregate metrics DataFrame
            metrics_df = format_evaluation_result(result)
            
            # Save to cache (questions, ground truth, and model outputs for display)
            cache_data = {
                "model_used": llm_config.google_model,
                "evaluator_model": llm_config.groq_model,
                "metrics_df": metrics_df.to_dict(orient="records") if isinstance(metrics_df, pd.DataFrame) else None,
                "evaluation_questions": user_inputs,
                "evaluation_references": references,
                "evaluation_responses": responses,
            }
            save_evaluation_cache(cache_data)
            
            # Display results
            st.success("✅ Evaluation completed and cached!")
            st.subheader("📊 Aggregate Evaluation Metrics")
            
            # Display aggregate metrics (already calculated in format_evaluation_result)
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)
            st.caption("📊 These are aggregate metrics (averages) across all evaluation questions.")
            
            # Show questions with ground truth and model output
            st.subheader("📋 Questions Used for Evaluation")
            for idx, question in enumerate(user_inputs, 1):
                with st.expander(f"{idx}. {question}", expanded=(idx == 1)):
                    st.markdown("**Ground truth query**")
                    st.code(references[idx - 1], language="sql")
                    st.markdown("**Model output query**")
                    st.code(responses[idx - 1], language="sql")
            st.info("💾 Results have been cached. Next time you can view them instantly using 'View Cached Results'.")
        
    except Exception as e:
        st.error(f"Failed to run evaluation: {str(e)}")
        st.info("This might be due to API issues or missing dependencies. Please check your configuration.")

# Footer
st.divider()
st.markdown(
    "<small>Built with Streamlit | Text-to-SQL powered by LangChain & Gemini</small>",
    unsafe_allow_html=True,
)
