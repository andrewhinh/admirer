"""Run validation test for question_answer module."""
import unittest

from question_answer.answer import Pipeline


class TestEvaluatePICa(unittest.TestCase):
    """Evaluate Caption on the additionally-added PICa examples."""

    def test_evaluate(self):
        pipeline = Pipeline()
        return pipeline.evaluate()


def main():
    testcase = TestEvaluatePICa()
    return testcase.test_evaluate()


if __name__ == "__main__":
    main()
