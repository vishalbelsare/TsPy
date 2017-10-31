ENV=".env"
PYTHON="python3"

clean:
	find . -name "*.pyc" -type f -delete
	find . -name "__pycache__" -type d -delete

remove:
	if [ -d $(ENV) ] ; then \
		rm -rf $(ENV) ; \
	fi ;

setup:
	make remove ;
	virtualenv -p $(PYTHON) $(ENV) ;
	source "./$(ENV)/bin/activate" && \
	pip install -r requirements.txt && \
	make test ;

test:
	if [ ! -d $(ENV) ]; then \
		make setup ; \
	fi ;
	source "./$(ENV)/bin/activate" && \
	pytest ;
	make clean ;