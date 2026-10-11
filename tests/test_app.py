import unittest

import app


class FlaskSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = app.app.test_client()

    def test_homepage_is_available(self):
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Course", response.data)

    def test_health_endpoint_is_available(self):
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), {"status": "ok"})

    def test_skills_endpoint_returns_skills(self):
        response = self.client.get("/api/skills")

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertIsInstance(payload["skills"], list)
        self.assertGreater(len(payload["skills"]), 0)

    def test_recommend_requires_json(self):
        response = self.client.post("/api/recommend")

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json()["error"], "No JSON body provided")

    def test_feedback_rejects_unknown_session(self):
        response = self.client.post(
            "/api/feedback",
            json={"session_id": "unknown", "course_index": 0, "feedback": "skip"},
        )

        self.assertEqual(response.status_code, 400)
