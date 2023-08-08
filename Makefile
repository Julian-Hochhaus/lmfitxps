all: init test

init:
	pip install -r requirements.txt

test: test_backgrounds test_models
test_backgrounds:
	py.test tests/test_backgrounds.py
test_models:
	py.test tests/test_models.py
.PHONY: init test
