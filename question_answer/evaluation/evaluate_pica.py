"""Run validation test for question_answer module."""
import sys
import unittest

sys.path.append("../")
from question_answer.answer import Pipeline


class TestEvaluateCaption(unittest.TestCase):
    """Evaluate Caption on the additionally-added PICa examples."""

    def test_evaluate(self):
        pipeline = Pipeline()
        return pipeline.evaluate()


def main():
    testcase = TestEvaluateCaption()
    return testcase.test_evaluate()


if __name__ == "__main__":
    main()
