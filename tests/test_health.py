from fastapi.testclient import TestClient

from research_assistant.main import app

client = TestClient(app)


def test_health_check() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "local-rag-research-assistant",
    }
