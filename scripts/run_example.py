"""
Example script: run a single Text-to-SQL query and print the result.
"""

from text_to_sql.db import get_database, run_query
from text_to_sql.llm_models import get_query_llm
from text_to_sql.chains import create_sql_chain, generate_sql_query


def main() -> None:
    db = get_database()
    llm = get_query_llm()
    sql_chain = create_sql_chain(db, llm)

    question = "What is the total 'Line Total' for Geiss Company"
    sql = generate_sql_query(sql_chain, question)

    print("Generated SQL query:")
    print(sql)

    result = run_query(db, sql)
    print("Query result:")
    print(result)


if __name__ == "__main__":
    main()

