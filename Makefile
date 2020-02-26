# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = docsrc
BUILDDIR      = docs

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# This calls sphinx-apidoc to create an api folder in SOURCEDIR with
# documentation for all Python modules
buildapi:
	@sphinx-apidoc -fMeET -o docsrc/api designer 
	@echo "Auto-generation of API documentation finished. " \
			"The generated files are in 'api/'"

github:
	@make html
	@cp -a docs/html/. ./docs
	@rm -r docs/html/

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
