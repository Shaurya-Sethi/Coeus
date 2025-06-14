import importlib.util
import os
import sys
import types

import pytest

# Provide lightweight stubs for optional dependencies used in ``app`` so that
# the module can be imported without installing them.
sys.modules.setdefault(
    "streamlit",
    types.SimpleNamespace(
        cache_data=lambda x: x,
        cache_resource=lambda x: x,
        sidebar=types.SimpleNamespace(
            header=lambda *a, **k: None,
            text_input=lambda *a, **k: "",
            slider=lambda *a, **k: 1,
        ),
        text_area=lambda *a, **k: "",
        button=lambda *a, **k: False,
        subheader=lambda *a, **k: None,
        json=lambda *a, **k: None,
        code=lambda *a, **k: None,
        success=lambda *a, **k: None,
        error=lambda *a, **k: None,
        write=lambda *a, **k: None,
        expander=lambda *a, **k: types.SimpleNamespace(),
    ),
)
sys.modules.setdefault(
    "neo4j",
    types.SimpleNamespace(
        GraphDatabase=types.SimpleNamespace(driver=lambda *a, **k: None)
    ),
)
sys.modules.setdefault(
    "sentence_transformers",
    types.SimpleNamespace(SentenceTransformer=lambda *a, **k: None),
)
sys.modules.setdefault(
    "sklearn",
    types.SimpleNamespace(),
)
metrics_stub = types.SimpleNamespace(
    pairwise=types.SimpleNamespace(cosine_similarity=lambda a, b: [[0]])
)
sys.modules.setdefault("sklearn.metrics", metrics_stub)
sys.modules.setdefault("sklearn.metrics.pairwise", metrics_stub.pairwise)
sys.modules.setdefault(
    "requests",
    types.SimpleNamespace(
        post=lambda *a, **k: types.SimpleNamespace(
            status_code=200, json=lambda: {"choices": [{"message": {"content": ""}}]}
        )
    ),
)

SPEC = importlib.util.spec_from_file_location(
    "app", os.path.join(os.path.dirname(__file__), "..", "app.py")
)
app = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(app)
ValidationAgent = app.ValidationAgent

sqlglot = pytest.importorskip("sqlglot")


@pytest.fixture
def agent():
    schema = {
        "users": {"columns": ["id", "name", "age"]},
        "orders": {"columns": ["user_id", "amount"]},
    }
    return ValidationAgent(schema)


def test_join_and_alias(agent):
    query = "SELECT u.id, o.amount FROM users AS u JOIN orders o ON u.id = o.user_id"
    tables, cols = agent.parse_query(query)
    assert set(tables) == {"users", "orders"}
    assert "id" in cols and "amount" in cols


def test_subquery_and_quoted_names(agent):
    query = (
        'SELECT "u"."name" FROM (SELECT * FROM "users") AS u '
        'WHERE u.id IN (SELECT "user_id" FROM orders)'
    )
    tables, cols = agent.parse_query(query)
    assert set(tables) == {"users", "orders"}
    assert {"name", "id", "user_id"}.issubset(set(cols))
