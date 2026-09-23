.PHONY: all figures verify

all: figures
	python3 tables/build_results.py
	python3 tables/build_baselines.py
	latexmk -pdf -no-shell-escape -interaction=nonstopmode -halt-on-error paper.tex

figures:
	python3 figs/data/extract_offsets.py
	cd figs && latexmk -pdf -no-shell-escape -interaction=nonstopmode -halt-on-error overview.tex timing.tex

verify: all
	python3 verify_submission.py
