import unittest

from fastapi.testclient import TestClient

from server.app import app
import server.app as server_app


class ServerApiTests(unittest.TestCase):
    def setUp(self):
        server_app.runtime_session = None
        self.client = TestClient(app)

    def test_health(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json().get("status"), "ok")

    def test_reset_state_step_flow(self):
        reset_response = self.client.post("/reset", json={})
        self.assertEqual(reset_response.status_code, 200)
        reset_payload = reset_response.json()
        self.assertIn("observation", reset_payload)
        self.assertIn("step_count", reset_payload)
        self.assertEqual(reset_payload["step_count"], 0)

        state_response = self.client.get("/state")
        self.assertEqual(state_response.status_code, 200)
        state_payload = state_response.json()
        self.assertTrue(state_payload.get("initialized"))

        step_response = self.client.post(
            "/step",
            json={"action": "edit_config: replace npm tset with npm test"},
        )
        self.assertEqual(step_response.status_code, 200)
        step_payload = step_response.json()
        self.assertIn("reward", step_payload)
        self.assertIn("done", step_payload)

    def test_step_requires_reset(self):
        server_app.runtime_session = None
        client = TestClient(app)
        response = client.post("/step", json={"action": "read_logs: inspect logs"})
        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
