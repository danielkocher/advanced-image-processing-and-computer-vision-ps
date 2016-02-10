FILENAME = report

all: default

default: $(FILENAME).tex
	pdflatex $(FILENAME).tex
	#bibtex $(FILENAME).aux
	pdflatex $(FILENAME).tex
	pdflatex $(FILENAME).tex

clean:
	rm -f *.log *.pdf *.aux *.bbl *.blg *.log