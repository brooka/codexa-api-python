from fastapi.testclient import TestClient

from codexa_api.main import create_app


def test_health_reports_the_loaded_packs(client):
    assert client.get("/health").json() == {"status": "ok", "packs": 2, "documents": 7, "ask_enabled": True}


def test_search_returns_hits(client):
    response = client.get("/search", params={"q": "Jesus wept", "mode": "keyword", "limit": 3})
    assert response.status_code == 200
    hit = response.json()["hits"][0]
    assert (hit["ref"], hit["label"], hit["type"]) == ("John 11:35", "KJV", "verse")


def test_search_rejects_bad_input(client):
    assert client.get("/search", params={"q": "grace", "mode": "fuzzy"}).status_code == 422
    assert client.get("/search", params={"q": ""}).status_code == 422


def test_ask_answers_with_citations(client):
    body = client.get("/ask", params={"q": "Who created the earth?"}).json()
    assert body["model"] == "fake-model"
    assert body["cited"] == [1]
    assert len(body["sources"]) == 3


def test_ask_is_unavailable_without_an_api_key(searcher):
    with TestClient(create_app(searcher, assistant=None)) as client:
        assert client.get("/ask", params={"q": "grace"}).status_code == 503


def test_ask_requires_the_key_when_one_is_set(searcher, assistant):
    with TestClient(create_app(searcher, assistant, ask_key="s3cret")) as client:
        assert client.get("/ask", params={"q": "grace"}).status_code == 401
        assert client.get("/ask", params={"q": "grace"}, headers={"X-API-Key": "s3cret"}).status_code == 200


def test_docs_and_schema_are_served(client):
    assert "<html" in client.get("/docs").text.lower()
    schema = client.get("/openapi.json").json()
    assert schema["components"]["securitySchemes"]["APIKeyHeader"]["name"] == "X-API-Key"
