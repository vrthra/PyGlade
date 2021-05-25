gen:
	python3 src/glade.py

fuzz:
	python3 src/fuzz.py grammar.json
