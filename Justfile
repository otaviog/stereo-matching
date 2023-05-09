about:
	@echo Common maintenance tasks

pylint:
	pylint stereomatch/ tests/

unit-tests:
	pytest tests/ --junitxml=junit/test-results.xml --cov=com --cov-report=xml --cov-report=html -k "not bm_"

benchmark:
	pytest tests -k bm_ --csv=benchmarks.csv
