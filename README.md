# Coeus: NL-2-SQL Using RAG & LLM

This application is a Streamlit-based tool that utilizes NLP and Neo4j to generate SQL queries dynamically. It fetches the schema from a Neo4j database, processes user queries using Sentence Transformers, and generates SQL queries via the ArliAI API.

## Directory Overview

- `e-commerce/` – Sample SQL schema and Cypher scripts for an e-commerce database. These files can be imported into Neo4j or a SQL engine to experiment with the query generator.
- `uber like/` – Cypher scripts describing a ride‑sharing schema similar to an Uber application. Useful for testing with a different domain.

## Features
- **Fetch Schema from Neo4j**: Extracts table and column descriptions along with relationships.
- **Sentence Transformer Embeddings**: Converts schema and user queries into vector embeddings.
- **Similarity-Based Schema Pruning**: Identifies relevant tables and columns based on query similarity.
- **Relationship-Aware Filtering**: Ensures key columns related to relationships are retained.
- **SQL Query Generation**: Uses ArliAI API to generate SQL queries based on the refined schema.
- **Configurable Similarity Threshold**: Allows fine-tuning relevance detection for query context.
- **Secure Credentials Input**: Uses password fields for sensitive information in Streamlit UI.

## Setup

1. **Install prerequisites**
   - Python 3.8 or newer
   - A running Neo4j instance
   - (Optional) Create a virtual environment

2. **Clone the repository**
   ```sh
   git clone https://github.com/your-username/Coeus.git
   cd Coeus
   ```

3. **Install dependencies**
   ```sh
   pip install -r requirements.txt
   ```

4. **Run the app**
   ```sh
   streamlit run app.py
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

### UI Overview
```
+--------------------------------------------+
| SQL Query Generator Using NLP              |
+--------------------------------------------+
| Sidebar                                    |
|  - Neo4j settings                          |
|  - API key                                 |
|  - Similarity slider                       |
+--------------------------------------------+
| Main Area                                  |
|  - User query input                        |
|  - Generate button                         |
|  - Schema and SQL results                  |
+--------------------------------------------+
```

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

## Contributing

Contributions are welcome! Please fork the repository and open a pull request with your changes.
By contributing you agree that your code will be released under the terms of the
[MIT License](LICENSE).

## License
This project is licensed under the [MIT License](LICENSE).

