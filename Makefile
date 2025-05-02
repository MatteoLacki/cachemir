make:
	echo "Welcome to Project 'cachemir'"

upload_test_pypi:
	rm -rf dist || True
	python setup.py sdist
	twine -r testpypi dist/* 

upload_pypi:
	rm -rf dist || True
	python setup.py sdist
	twine upload dist/* 

ve_cachemir:
	python3 -m venv ve_cachemir
