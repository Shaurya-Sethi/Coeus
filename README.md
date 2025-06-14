# Coeus: NL-2-SQL Using RAG & LLM

This application is a Streamlit-based tool that utilizes NLP and Neo4j to generate SQL queries dynamically. It fetches the schema from a Neo4j database, processes user queries using Sentence Transformers, and generates SQL queries via the ArliAI API.

## Directory Overview

* `e-commerce/` – Sample SQL schema and Cypher scripts for an e-commerce database. These files can be imported into Neo4j or a SQL engine to experiment with the query generator.
* `uber like/` – Cypher scripts describing a ride‑sharing schema similar to an Uber application. Useful for testing with a different domain.

## Features

* **Fetch Schema from Neo4j**: Extracts table and column descriptions along with relationships.
* **Sentence Transformer Embeddings**: Converts schema and user queries into vector embeddings.
* **Similarity-Based Schema Pruning**: Identifies relevant tables and columns based on query similarity.
* **Relationship-Aware Filtering**: Ensures key columns related to relationships are retained.
* **SQL Query Generation**: Uses ArliAI API to generate SQL queries based on the refined schema.
* **Configurable Similarity Threshold**: Allows fine-tuning relevance detection for query context.
* **Environment-Based Secrets**: Credentials are loaded from `.env` or `st.secrets` with a UI fallback for local development.

## Setup

1. **Install prerequisites**

   * Python 3.8 or newer
   * A running Neo4j instance
   * (Optional) Create a virtual environment

2. **Clone the repository**

   ```sh
   git clone https://github.com/Abhay-Sastha-S/Coeus.git
   cd Coeus
   ```

3. **Install dependencies**
   Install the required packages from the `requirements.txt` file at the repository root:

   ```sh
   pip install -r requirements.txt
   ```
4. **Set up pre-commit hooks**
   Install `pre-commit` and initialize the git hooks:

   ```sh
   pip install pre-commit
   pre-commit install
   ```


5. **Create a `.env` file**
   Define the following variables (or use `st.secrets` in a deployed environment):

   ```dotenv
   NEO4J_URI=bolt://localhost:7687
   NEO4J_USER=neo4j
   NEO4J_PASSWORD=your_password
   NEO4J_DATABASE=e-commerce
   ARLIAI_API_KEY=your_api_key
   ```

6. **Run the app**

   ```sh
   streamlit run app.py
   ```

## Usage

### Running the Application

```sh
streamlit run app.py
```

### Configuring Inputs

Set the required variables in a `.env` file or `st.secrets` as shown above. If a
variable is missing, the app will display a text input in the sidebar so you can
provide it manually during local development.

* **Neo4j URI** – connection string for Neo4j (e.g., `bolt://localhost:7687`).
* **Neo4j User & Password** – authentication credentials.
* **Database Name** – the Neo4j database containing the schema.
* **ArliAI API Key** – required to generate SQL queries.
* **Similarity Threshold** – adjusts the sensitivity for schema relevance detection.
* **User Query** – natural language input describing the desired SQL query.

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

* Connects to Neo4j.
* Fetches table and column schema.
* Retrieves table relationships.

### `fetch_embedding_locally(text)`

* Generates vector embeddings for schema and queries.

### `generate_schema_embeddings_locally(schema)`

* Converts the entire schema into vector embeddings.

### `find_relevant_schema(query_embedding, schema_embeddings, threshold)`

* Identifies tables relevant to the user query based on embedding similarity.

### `identify_relationship_columns(schema, relationships)`

* Extracts columns related to inter-table relationships.

### `prune_non_relationship_columns(user_query_embedding, schema, relationships, threshold)`

* Filters out unrelated columns while retaining key relationship attributes.

### `generate_sql_arliAI(api_key, user_query, pruned_schema)`

* Uses the ArliAI API to generate SQL queries based on a refined schema.

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

* **Connection Issues**: Ensure the Neo4j instance is running and accessible.
* **Embedding Errors**: Verify that `sentence-transformers` is installed and working.
* **API Failure**: Check if the ArliAI API key is valid and the service is online.
* **Incorrect SQL Queries**: Adjust the similarity threshold or refine column selection.

## Contributing

Contributions are welcome! Please fork the repository and open a pull request with your changes.
By contributing you agree that your code will be released under the terms of the
[MIT License](LICENSE).

## License

This project is licensed under the [MIT License](LICENSE).
