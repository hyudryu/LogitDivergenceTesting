from fastapi.testclient import TestClient

from app.main import app


def test_health_and_suites():
    with TestClient(app) as client:
        assert client.get("/api/health").json() == {"ok": True}
        suites = client.get("/api/suites").json()
        assert "divergence" in suites


def test_create_run_validates_endpoint_count():
    with TestClient(app) as client:
        r = client.post(
            "/api/runs",
            json={"endpoints": [{"name": "a", "base_url": "http://localhost:9"}]},
        )
        assert r.status_code == 422


def test_missing_run_404():
    with TestClient(app) as client:
        assert client.get("/api/runs/doesnotexist").status_code == 404
