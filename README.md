# NL-2-SQL Using RAG & LLM

This application is a Streamlit-based tool that utilizes NLP and Neo4j to generate SQL queries dynamically. It fetches the schema from a Neo4j database, processes user queries using Sentence Transformers, and generates SQL queries via the ArliAI API.

## Features
- **Fetch Schema from Neo4j**: Extracts table and column descriptions along with relationships.
- **Sentence Transformer Embeddings**: Converts schema and user queries into vector embeddings.
- **Similarity-Based Schema Pruning**: Identifies relevant tables and columns based on query similarity.
- **Relationship-Aware Filtering**: Ensures key columns related to relationships are retained.
- **SQL Query Generation**: Uses ArliAI API to generate SQL queries based on the refined schema.
- **Configurable Similarity Threshold**: Allows fine-tuning relevance detection for query context.
- **Secure Credentials Input**: Uses password fields for sensitive information in Streamlit UI.

## Installation

### Prerequisites
Ensure you have the following installed:
- Python 3.8+
- Neo4j (Running instance with a valid database schema)
- Virtual environment (optional but recommended)

### Clone the Repository
```sh
$ git clone https://github.com/your-repo/sql-query-generator.git
$ cd sql-query-generator
```

### Create a Virtual Environment (Optional but Recommended)
```sh
$ python -m venv venv
$ source venv/bin/activate  # On macOS/Linux
$ venv\Scripts\activate    # On Windows
```

### Install Dependencies
```sh
$ pip install -r requirements.txt
```

## Usage

### Running the Application
```sh
$ streamlit run app.py
```

### Configuring Inputs
- **Neo4j URI**: The connection string for Neo4j (e.g., `bolt://localhost:7687`).
- **Neo4j User & Password**: Authentication credentials.
- **Database Name**: The Neo4j database containing the schema.
- **ArliAI API Key**: Required to generate SQL queries.
- **Similarity Threshold**: Adjusts the sensitivity for schema relevance detection.
- **User Query**: Natural language input describing the desired SQL query.

### Workflow
1. The schema is fetched from Neo4j.
2. The schema is converted into vector embeddings.
3. The user inputs a query.
4. The system identifies relevant schema elements.
5. Relationship-aware pruning refines column selection.
6. A SQL query is generated via the ArliAI API.
7. The final query is displayed in the Streamlit UI.

## Functionality Overview

### `Neo4jHandler`
- Connects to Neo4j.
- Fetches table and column schema.
- Retrieves table relationships.

### `fetch_embedding_locally(text)`
- Generates vector embeddings for schema and queries.

### `generate_schema_embeddings_locally(schema)`
- Converts the entire schema into vector embeddings.

### `find_relevant_schema(query_embedding, schema_embeddings, threshold)`
- Identifies tables relevant to the user query based on embedding similarity.

### `identify_relationship_columns(schema, relationships)`
- Extracts columns related to inter-table relationships.

### `prune_non_relationship_columns(user_query_embedding, schema, relationships, threshold)`
- Filters out unrelated columns while retaining key relationship attributes.

### `generate_sql_arliAI(api_key, user_query, pruned_schema)`
- Uses the ArliAI API to generate SQL queries based on a refined schema.

## Example

### Input
```plaintext
User Query: List all users, their orders, and the products in those orders.
```

### Output
```sql
SELECT users.id, users.name, orders.id AS order_id, orders.date, products.name AS product_name
FROM users
JOIN orders ON users.id = orders.user_id
JOIN order_products ON orders.id = order_products.order_id
JOIN products ON order_products.product_id = products.id;
```

## Troubleshooting

- **Connection Issues**: Ensure the Neo4j instance is running and accessible.
- **Embedding Errors**: Verify that `sentence-transformers` is installed and working.
- **API Failure**: Check if the ArliAI API key is valid and the service is online.
- **Incorrect SQL Queries**: Adjust the similarity threshold or refine column selection.

## License
This project is licensed under the MIT License.

