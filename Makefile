all: init test

init:
	pip install -r requirements.txt

test: test_backgrounds test_models
test_backgrounds:
	py.test tests/test_backgrounds.py
test_models:
	py.test tests/test_models.py
logo:
	python3 examples/logo.py
examples: te
te:
	PYTHONPATH=$$PYTHONPATH:$$(pwd) python3 examples/tougaard_extend.py
te2:
	PYTHONPATH=$$PYTHONPATH:$$(pwd) python3 examples/tougaard_extend2.py
.PHONY: init test logo
