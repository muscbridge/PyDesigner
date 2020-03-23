# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

# Added to auto build API doc during make
APIBUILD	  ?= sphinx-apidoc
APIBUILDDIR   = source/api
APIDIR        = ../designer

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
# This calls sphinx-apidoc to create an api folder in SOURCEDIR with
# documentation for all Python modules
	@rm -rfv "$(BUILDDIR)"/*
	@$(APIBUILD) -fMEeT -o "$(APIBUILDDIR)" "$(APIDIR)"
	@echo "Auto-generation of API documentation finished. " \
			"The generated files are in 'api/'"
# Then build
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
