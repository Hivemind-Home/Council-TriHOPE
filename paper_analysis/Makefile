# Regenerate every number, table and figure the paper prints from the run data in data/.
PY ?= python

.PHONY: all results figures
all: results figures

results:  ## tables/*.tex + tables/numbers.tex
	mkdir -p tables figures
	$(PY) scripts/build_results.py --final

figures:  ## figures/*.pdf (needs a LaTeX install for text rendering)
	mkdir -p figures
	$(PY) scripts/make_figures.py
