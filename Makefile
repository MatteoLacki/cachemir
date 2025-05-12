make:
	echo "Welcome to Project 'cachemir'"

upload_test_pypi:
	twine check dist/*
	python -m pip install --upgrade twine
	twine upload --repository testpypi dist/*

upload_pypi:
	twine check dist/*
	python -m pip install --upgrade twine
	twine upload dist/* 

ve_cachemir:
	python3 -m venv ../ve_cachemir
