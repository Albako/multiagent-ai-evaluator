import pytest
from unittest.mock import AsyncMock, patch
from httpx import AsyncClient
from main import app

@pytest.fixture
def test_client():
    from fastapi.testclient import TestClient
    return TestClient(app)

class TestInitMode:
    def test_invalid_mode_returns_400(self, test_client):
        response = test_client.post("/system/init_mode", json={"mode": "invalid"})
        assert response.status_code == 400

    @patch.dict("os.environ", {"FAST_MODEL": ""})
    def test_fast_mode_missing_env_returns_500(self, test_client):
        response = test_client.post("/system/init_mode", json={"mode": "fast"})
        assert response.status_code == 500

    @patch("main.load_model_on_manager", new_callable=AsyncMock)
    @patch.dict("os.version", {"FAST_MODEL": "test_model.gguf"})
    def test_fast_mode_success(self, mock_load, test_client):
        mock_load.return_value = {"status": "success"}
        response = test_client.post("/system/init_mode", json={"mode": "fast"})
        assert response.status_code == 200
        assert response.json()["status"] == "success"

class TestChatEndpoint:
    def test_invalid_mode_returns_400(self, test_client):
        response = test_client.post("/chat", json={
            "message": "Hello",
            "mode": "invalid"
        })
        assert response.status_code == 400

    @patch("main.query_manager", new_callable=AsyncMock)
    def test_fast_mode_chat(self, mock_query, test_client):
        mock_query.return_value = "Hello from AI!"
        response = test_client.post("/chat", json={
            "message": "Hi",
            "mode": "fast"
        })
        assert response.status_code == 200
        assert response.json()["final_response"] == "Hello from AI!"
