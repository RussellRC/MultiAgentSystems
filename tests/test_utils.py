from pydantic_evals.reporting import EvaluationReport


def eval_report_cases(report: EvaluationReport):
    for case_result in report.cases:
        assert_eval_results(case_result)

def assert_eval_results(case_result):
    """Assert all scores and assertions on a single case result passed."""
    for score_name, score in case_result.scores.items():
        assert score.value, f"{score_name} failed for {case_result.name}"
    for assertion_name, assertion in case_result.assertions.items():
        assert assertion.value, f"{assertion_name} failed for {case_result.name}"
