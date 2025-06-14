"""Streamlit app for generating and validating SQL queries from natural language.

The application connects to a Neo4j database to obtain table schemas, generates
embeddings using SentenceTransformer, and leverages ArliAI to create SQL queries
based on user input. Generated queries are validated against the schema and can
optionally be corrected through the validation agent.
"""

import streamlit as st
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json
import requests
import re

# Neo4j Handler
class Neo4jHandler:
    def __init__(self, uri, user, password, database):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database

    def close(self):
        self.driver.close()

    def fetch_table_schema(self):
        query = (
            "MATCH (t:Table)<-[:BELONGS_TO]-(c:Column)\n"
            "RETURN t.name AS table, t.description AS table_description, "
            "c.name AS column, c.description AS column_description"
        )
        with self.driver.session(database=self.database) as session:
            result = session.run(query)
            schema = {}
            for record in result:
                table = record["table"]
                if table not in schema:
                    schema[table] = {"description": record["table_description"], "columns": []}
                schema[table]["columns"].append({
                    "name": record["column"],
                    "description": record["column_description"]
                })
            return schema

    def fetch_relationships(self, relevant_tables):
        """
        Fetch the relationships between the relevant tables.
        """
        relationships = {}
        with self.driver.session(database=self.database) as session:
            for table in relevant_tables:
                query = f"""
                MATCH (t:{table})-[r]->(related)
                RETURN type(r) AS relationship, related
                """
                result = session.run(query)
                related_tables = {}
                for record in result:
                    rel_type = record["relationship"]
                    related_table = record["related"]
                    if related_table not in related_tables:
                        related_tables[related_table] = []
                    related_tables[related_table].append({"type": rel_type})
                relationships[table] = related_tables
        return relationships

# Sentence-Transformer Model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

@st.cache_data
def fetch_embedding_locally(text):
    """
    Generate embedding for a given text using SentenceTransformer.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"Input must be a non-empty string. Received: {text}")

    # Generate embedding
    embedding = model.encode([text])[0]
    return embedding.tolist()

@st.cache_data
def generate_schema_embeddings_locally(schema):
    """
    Generate schema embeddings using local SentenceTransformer with descriptions.
    """
    schema_embeddings = {}
    for table, data in schema.items():
        # Create context that includes both the table and column descriptions
        table_context = f"Table: {table}, Description: {data['description']}, Columns: "
        column_contexts = [
            f"{column['name']} (Description: {column['description']})"
            for column in data["columns"]
        ]
        full_context = table_context + ", ".join(column_contexts)

        try:
            embedding = fetch_embedding_locally(full_context)
            schema_embeddings[table] = {
                "embedding": embedding,
                "columns": [col["name"] for col in data["columns"]],
            }
        except Exception as e:
            print(f"Failed to generate embedding for {table}. Error: {e}")
            raise
    return schema_embeddings

@st.cache_data
def find_relevant_schema(
    query_embedding,
    schema_embeddings,
    top_k_tables=5,
    similarity_threshold=None,
):
    """Select relevant tables using either Top-K or threshold filtering.

    Args:
        query_embedding (List[float]): Embedding of the user query.
        schema_embeddings (dict): Table embeddings generated from the schema.
        top_k_tables (int, optional): Number of top tables to return based on
            similarity. Used when ``similarity_threshold`` is ``None``.
        similarity_threshold (float, optional): If provided, tables with
            similarity above this threshold will be returned instead of using
            ``top_k_tables``.

    Returns:
        dict: Mapping of table names to their list of columns.
    """

    table_similarities = []
    for table, data in schema_embeddings.items():
        similarity = cosine_similarity([query_embedding], [data["embedding"]])[0][0]
        table_similarities.append((table, similarity))

    if similarity_threshold is not None:
        return {
            table: schema_embeddings[table]["columns"]
            for table, sim in table_similarities
            if sim > similarity_threshold
        }

    # Fallback to Top-K selection
    table_similarities.sort(key=lambda x: x[1], reverse=True)
    selected = table_similarities[:top_k_tables]
    return {table: schema_embeddings[table]["columns"] for table, _ in selected}

def generate_sql_arliAI(api_key, user_query, pruned_schema):
    """
    Generate SQL query using ArliAI API.
    """
    schema_context = "\n".join(
        [
            f"- Table: {table} (Columns: {', '.join(columns)})"
            for table, columns in pruned_schema.items()
        ]
    )
    prompt = f"""
    Use the provided schema to generate a valid SQL query.

    Schema:
    {schema_context}

    User Query: {user_query}
    SQL:
    """

    payload = json.dumps({
        "model": "Mistral-Nemo-12B-Instruct-2407",
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant for generating SQL queries.",
            },
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "top_p": 0.9,
        "max_tokens": 1024,
        "n": 1
    })

    headers = {
        'Content-Type': 'application/json',
        'Authorization': f"Bearer {api_key}"
    }

    response = requests.post(
        "https://api.arliai.com/v1/chat/completions",
        headers=headers,
        data=payload,
    )

    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"].strip()
    else:
        raise Exception(f"ArliAI API call failed: {response.text}")

@st.cache_data
def identify_relationship_columns(schema, relationships):
    """
    Identify relationship columns based on relationships between tables.
    """
    relationship_columns = set()  # Use a set to avoid duplicates

    # For each relationship, find the columns that define the relationship
    for from_table, related_tables in relationships.items():
        for to_table, rels in related_tables.items():
            for rel in rels:
                for key in rel.keys():  # Relationship property/column
                    relationship_columns.add(key)  # Add property/column from the relationship
    return relationship_columns

@st.cache_data
def prune_non_relationship_columns(
    user_query_embedding,
    schema,
    relationships,
    top_k_cols=5,
    similarity_threshold=None,
):
    """Prune columns by keeping relationship columns and Top-K most relevant.

    Args:
        user_query_embedding (List[float]): Embedding of the user query.
        schema (dict): Full database schema with descriptions.
        relationships (dict): Relationship information between tables.
        top_k_cols (int, optional): Number of most similar columns to keep per
            table when ``similarity_threshold`` is ``None``.
        similarity_threshold (float, optional): If provided, keep columns with
            similarity above this value instead of using ``top_k_cols``.

    Returns:
        dict: Mapping of table names to a list of retained column names.
    """

    pruned_schema = {}
    relationship_columns = identify_relationship_columns(schema, relationships)

    # Keep track of all unique columns across tables
    unique_columns = set()

    for table, table_data in schema.items():
        relevant_columns = []
        column_scores = []

        for column in table_data["columns"]:
            column_name = column["name"]

            # If the column is part of relationships, keep it immediately
            if column_name in relationship_columns:
                if column_name not in unique_columns:
                    relevant_columns.append(column_name)
                    unique_columns.add(column_name)
                continue

            column_context = (
                f"Table: {table}, Column: {column_name}, "
                f"Description: {table_data['description']}"
            )
            column_embedding = fetch_embedding_locally(column_context)
            column_similarity = cosine_similarity(
                [user_query_embedding], [column_embedding]
            )[0][0]

            column_scores.append((column_name, column_similarity))

        if similarity_threshold is not None:
            filtered = [
                name
                for name, score in column_scores
                if score > similarity_threshold and name not in unique_columns
            ]
        else:
            column_scores.sort(key=lambda x: x[1], reverse=True)
            filtered = [
                name
                for name, _ in column_scores[:top_k_cols]
                if name not in unique_columns
            ]

        relevant_columns.extend(filtered)
        unique_columns.update(filtered)

        # Only add the table if it has relevant columns
        if relevant_columns:
            pruned_schema[table] = relevant_columns

    return pruned_schema

class ValidationAgent:
    def __init__(self, schema):
        self.schema = schema

    def parse_query(self, query):
        """
        Parse the SQL query to identify tables and columns.

        This simple regex-based approach works for basic queries but will not
        handle nested statements or advanced SQL syntax.
        """
        tables = re.findall(r'FROM\s+(\w+)', query, re.IGNORECASE)
        columns = re.findall(r'SELECT\s+(.*?)\s+FROM', query, re.IGNORECASE)
        columns = [col.strip() for col in columns[0].split(",")] if columns else []
        return tables, columns

    def validate_schema(self, tables, columns):
        """
        Validate that the tables and columns in the query exist in the schema.
        """
        errors = []
        for table in tables:
            if table not in self.schema:
                errors.append(f"Table '{table}' does not exist in the schema.")
            else:
                for column in columns:
                    if column not in self.schema[table]["columns"]:
                        errors.append(f"Column '{column}' does not exist in table '{table}'.")
        return errors

    def correct_query(self, api_key, query, errors):
        """
        Send the erroneous SQL query and errors to the LLM for correction.
        """
        prompt = f"""
        The following SQL query has errors:

        Query:
        {query}

        Errors:
        {errors}

        Please correct the SQL query while ensuring it adheres to the schema.
        """

        payload = json.dumps(
            {
                "model": "Mistral-Nemo-12B-Instruct-2407",
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a helpful assistant for generating and "
                            "correcting SQL queries. give only sql query, do"
                            " not explain"
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.7,
                "top_p": 0.9,
                "max_tokens": 1024,
                "n": 1,
            }
        )

        headers = {
            'Content-Type': 'application/json',
            'Authorization': f"Bearer {api_key}"
        }

        response = requests.post(
            "https://api.arliai.com/v1/chat/completions",
            headers=headers,
            data=payload,
        )

        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"].strip()
        else:
            raise Exception(f"ArliAI API call failed: {response.text}")


def main():
    st.title("SQL Query Generator Using NLP with Validation")

    # Inputs
    st.sidebar.header("Configuration")
    neo4j_uri = st.sidebar.text_input("Neo4j URI", "bolt://localhost:7687")
    neo4j_user = st.sidebar.text_input("Neo4j User", "neo4j")
    neo4j_password = st.sidebar.text_input("Neo4j Password", type="password")
    database_name = st.sidebar.text_input("Database Name", "e-commerce")
    arliAI_api_key = st.sidebar.text_input("ArliAI API Key", type="password")

    # Sliders for Top-K control
    top_k_tables = st.sidebar.slider(
        "Top K Tables",
        min_value=1,
        max_value=10,
        value=5,
        step=1,
        help="Number of most relevant tables to select",
    )

    top_k_columns = st.sidebar.slider(
        "Top K Columns",
        min_value=1,
        max_value=20,
        value=5,
        step=1,
        help="Number of columns to keep per selected table",
    )

    user_query = st.text_area(
        "User Query",
        "List all users, their orders, and the products in those orders."
    )

    @st.cache_resource
    def initialize_schema_and_embeddings():
        """
        Fetch schema from Neo4j and generate embeddings.
        """
        neo4j_handler = Neo4jHandler(
            uri=neo4j_uri,
            user=neo4j_user,
            password=neo4j_password,
            database=database_name,
        )
        schema = neo4j_handler.fetch_table_schema()
        neo4j_handler.close()
        schema_embeddings = generate_schema_embeddings_locally(schema)
        return schema, schema_embeddings

    # Load schema and embeddings
    schema, schema_embeddings = initialize_schema_and_embeddings()

    st.subheader("Database Schema")
    st.json(schema)

    # Initialize the Neo4j handler again for fetching relationships after pruning
    neo4j_handler = Neo4jHandler(
        uri=neo4j_uri,
        user=neo4j_user,
        password=neo4j_password,
        database=database_name,
    )

    # Initialize Validation Agent
    validation_agent = ValidationAgent(schema)

    # Process Query
    if st.button("Generate SQL Query"):
        st.write("Processing your query...")

        try:
            user_query_embedding = fetch_embedding_locally(user_query)

            # Step 1: Select most relevant tables using Top-K
            relevant_tables = find_relevant_schema(
                user_query_embedding,
                schema_embeddings,
                top_k_tables=top_k_tables,
            )

            # Step 2: Fetch relationships for the pruned tables
            relationships = neo4j_handler.fetch_relationships(relevant_tables)

            # Step 3: Prune columns based on relationships and Top-K relevance
            pruned_schema = prune_non_relationship_columns(
                user_query_embedding,
                schema,
                relationships,
                top_k_cols=top_k_columns,
            )

            st.subheader("Pruned Schema")
            st.json(pruned_schema)

            # Generate SQL query based on pruned schema
            generated_sql = generate_sql_arliAI(
                arliAI_api_key,
                user_query,
                pruned_schema,
            )
            st.subheader("Generated SQL Query")
            st.code(generated_sql, language="sql")

            # Validate the generated SQL query
            tables, columns = validation_agent.parse_query(generated_sql)
            errors = validation_agent.validate_schema(tables, columns)

            if errors:
                st.error("Validation Errors Found:")
                for error in errors:
                    st.error(error)

                # Send the erroneous query and errors to the LLM for correction
                corrected_query = validation_agent.correct_query(
                    arliAI_api_key,
                    generated_sql,
                    errors,
                )

                st.subheader("Corrected SQL Query")
                st.code(corrected_query, language="sql")
            else:
                st.success("Generated SQL query is valid and aligns with the schema.")


        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
