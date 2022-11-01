import json

from locust import constant, HttpUser, task


image_url = "question_answer/tests/support/images/img.jpg"
question = "What color is my hair"


class AdmirerUser(HttpUser):
    """
    Simulated AWS Lambda User
    """

    wait_time = constant(1)
    headers = {"Content-type": "application/json"}
    payload = json.dumps({"image_url": image_url, "question": question})

    @task
    def predict(self):
        response = self.client.post("/", data=self.payload, headers=self.headers)
        pred = response.json()["pred"]
