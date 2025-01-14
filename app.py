import streamlit as st
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json
import requests

# Neo4j Handler
class Neo4jHandler:
    def __init__(self, uri, user, password, database):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database

    def close(self):
        self.driver.close()

    def fetch_table_schema(self):
        query = """
        MATCH (t:Table)<-[:BELONGS_TO]-(c:Column)
        RETURN t.name AS table, t.description AS table_description, c.name AS column, c.description AS column_description
        """
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

# Sentence-Transformer Model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def fetch_embedding_locally(text):
    """
    Generate embedding for a given text using SentenceTransformer.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"Input must be a non-empty string. Received: {text}")
    
    # Generate embedding
    embedding = model.encode([text])[0]
    return embedding.tolist()

def generate_schema_embeddings_locally(schema):
    """
    Generate schema embeddings using local SentenceTransformer with descriptions.
    """
    schema_embeddings = {}
    for table, data in schema.items():
        # Create context that includes both the table and column descriptions
        table_context = f"Table: {table}, Description: {data['description']}, Columns: "
        column_contexts = [f"{column['name']} (Description: {column['description']})" for column in data["columns"]]
        full_context = table_context + ", ".join(column_contexts)
        
        try:
            embedding = fetch_embedding_locally(full_context)
            schema_embeddings[table] = {"embedding": embedding, "columns": [column["name"] for column in data["columns"]]}
        except Exception as e:
            print(f"Failed to generate embedding for {table}. Error: {e}")
            raise
    return schema_embeddings

def find_relevant_schema(query_embedding, schema_embeddings, threshold=0.2):
    """
    Find tables in the schema with embeddings similar to the query embedding.
    """
    relevant_schema = {}
    for table, data in schema_embeddings.items():
        similarity = cosine_similarity([query_embedding], [data["embedding"]])[0][0]
        if similarity > threshold:
            relevant_schema[table] = data["columns"]
    return relevant_schema

def generate_sql_arliAI(api_key, user_query, pruned_schema):
    """
    Generate SQL query using ArliAI API.
    """
    schema_context = "\n".join(
        [f"- Table: {table} (Columns: {', '.join(columns)})" for table, columns in pruned_schema.items()]
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
            {"role": "system", "content": "You are a helpful assistant for generating SQL queries."},
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

    response = requests.post("https://api.arliai.com/v1/chat/completions", headers=headers, data=payload)

    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"].strip()
    else:
        raise Exception(f"ArliAI API call failed: {response.text}")

# Streamlit Application
def main():
    st.title("SQL Query Generator Using NLP")

    # Inputs
    st.sidebar.header("Configuration")
    neo4j_uri = st.sidebar.text_input("Neo4j URI", "bolt://localhost:7687")
    neo4j_user = st.sidebar.text_input("Neo4j User", "neo4j")
    neo4j_password = st.sidebar.text_input("Neo4j Password", type="password")
    database_name = st.sidebar.text_input("Database Name", "e-commerce")
    arliAI_api_key = st.sidebar.text_input("ArliAI API Key", type="password")

    user_query = st.text_area("User Query", "List all users, their orders, and the products in those orders.")

    @st.cache_data
    def initialize_schema_and_embeddings():
        """
        Fetch schema from Neo4j and generate embeddings.
        """
        neo4j_handler = Neo4jHandler(uri=neo4j_uri, user=neo4j_user, password=neo4j_password, database=database_name)
        schema = neo4j_handler.fetch_table_schema()
        neo4j_handler.close()
        schema_embeddings = generate_schema_embeddings_locally(schema)
        return schema, schema_embeddings

    # Load schema and embeddings
    schema, schema_embeddings = initialize_schema_and_embeddings()

    st.subheader("Database Schema")
    st.json(schema)

    # Process Query
    if st.button("Generate SQL Query"):
        st.write("Processing your query...")
        try:
            user_query_embedding = fetch_embedding_locally(user_query)
            pruned_schema = find_relevant_schema(user_query_embedding, schema_embeddings)
            st.subheader("Pruned Schema")
            st.json(pruned_schema)

            generated_sql = generate_sql_arliAI(arliAI_api_key, user_query, pruned_schema)
            st.subheader("Generated SQL Query")
            st.code(generated_sql, language="sql")
        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
