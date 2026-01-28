"""
Script to run RAGAS evaluation over a set of Text-to-SQL queries.
"""

from text_to_sql.chains import batch_generate_sql_queries, create_sql_chain
from text_to_sql.db import get_database, get_schema
from text_to_sql.evaluation import (
    build_evaluation_dataset,
    build_evaluation_components,
    run_ragas_evaluation,
)
from text_to_sql.llm_models import get_query_llm


def main() -> None:
    db = get_database()
    llm = get_query_llm()
    sql_chain = create_sql_chain(db, llm)

    schema_context = get_schema(db)
    retrieved_contexts = [schema_context]

    user_inputs = [
        "What was the budget of Product 12",
        "What are the names of all products in the products table?",
        "List all customer names from the customers table.",
        "Find the name and state of all regions in the regions table.",
        "What is the name of the customer with Customer Index = 1",
    ]

    responses = batch_generate_sql_queries(sql_chain, user_inputs)

    references = [
        "SELECT `2017 Budgets` FROM `2017_budgets` WHERE `Product Name` = 'Product 12';",
        "SELECT `Product Name` FROM products;",
        "SELECT `Customer Names` FROM customers;",
        "SELECT name, state FROM regions;",
        "SELECT `Customer Names` FROM customers WHERE `Customer Index` = 1;",
    ]

    dataset = build_evaluation_dataset(
        user_inputs=user_inputs,
        retrieved_contexts=retrieved_contexts,
        responses=responses,
        references=references,
    )

    components = build_evaluation_components()
    result = run_ragas_evaluation(dataset, components)

    print(result)


if __name__ == "__main__":
    main()

