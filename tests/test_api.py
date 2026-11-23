
from fastapi.testclient import TestClient

from app.main import app
from app.providers.mock import MockProvider


client = TestClient(app)


def test_health():
    """测试健康检查接口"""

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_list_models():
    """测试模型列表"""

    response = client.get("/v1/models")

    assert response.status_code == 200

    data = response.json()

    assert data["object"] == "list"
    assert data["data"][0]["id"] == "mock-chat"


def test_chat_completion():
    """测试正常模型调用"""

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "mock-chat",
            "messages": [
                {
                    "role": "user",
                    "content": "你好"
                }
            ]
        }
    )

    assert response.status_code == 200

    data = response.json()

    assert data["model"] == "mock-chat"
    assert data["object"] == "chat.completion"
    assert data["choices"][0]["message"]["role"] == "assistant"
    assert "Mock 模型收到" in data["choices"][0]["message"]["content"]

    assert data["usage"]["total_tokens"] == 0


def test_unknown_model():
    """测试不存在的模型"""

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "unknown-model",
            "messages": [
                {
                    "role": "user",
                    "content": "你好"
                }
            ]
        }
    )

    assert response.status_code == 404


def test_empty_messages():
    """测试空消息列表"""

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "mock-chat",
            "messages": []
        }
    )

    assert response.status_code == 422


def test_stream_not_supported():
    """测试当前不支持流式输出"""

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "mock-chat",
            "messages": [
                {
                    "role": "user",
                    "content": "你好"
                }
            ],
            "stream": True
        }
    )

    assert response.status_code == 400


def test_provider_exception(monkeypatch):
    """测试 Provider 内部异常"""

    async def mock_failure(self, request):
        raise RuntimeError("Simulated provider failure")

    monkeypatch.setattr(
        MockProvider,
        "chat_completion",
        mock_failure
    )

    test_client = TestClient(
        app,
        raise_server_exceptions=False
    )

    response = test_client.post(
        "/v1/chat/completions",
        json={
            "model": "mock-chat",
            "messages": [
                {
                    "role": "user",
                    "content": "你好"
                }
            ]
        }
    )

    assert response.status_code == 500
