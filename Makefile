gen:
	python3 src/glade.py  '(12324)'

fuzz:
	python3 src/fuzz.py grammar.json
