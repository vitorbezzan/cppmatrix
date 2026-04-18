.DEFAULT_GOAL := help

.PHONY: help
help:
	@printf "%s\n" \
		"Targets:" \
		"  format        Format C/C++ sources with astyle" \
		"  format-check  Check formatting (no changes)"

.PHONY: format format-check

ASTYLE ?= astyle
ASTYLE_CONFIG ?= .astylerc

format:
	@command -v "$(ASTYLE)" >/dev/null 2>&1 || { echo "error: astyle not found (install it and re-run)"; exit 127; }
	@files="$$(git ls-files -- '*.c' '*.cc' '*.cpp' '*.cxx' '*.h' '*.hh' '*.hpp' '*.hxx' | while IFS= read -r f; do test -s "$$f" && printf '%s\n' "$$f"; done)"; \
	test -n "$$files" || { echo "No non-empty C/C++ files found to format."; exit 0; }; \
	$(ASTYLE) --options="$(ASTYLE_CONFIG)" --suffix=none $$files

format-check:
	@command -v "$(ASTYLE)" >/dev/null 2>&1 || { echo "error: astyle not found (install it and re-run)"; exit 127; }
	@files="$$(git ls-files -- '*.c' '*.cc' '*.cpp' '*.cxx' '*.h' '*.hh' '*.hpp' '*.hxx' | while IFS= read -r f; do test -s "$$f" && printf '%s\n' "$$f"; done)"; \
	test -n "$$files" || { echo "No non-empty C/C++ files found to check."; exit 0; }; \
	output="$$( $(ASTYLE) --options="$(ASTYLE_CONFIG)" --suffix=none --dry-run --formatted $$files )"; \
	printf "%s\n" "$$output"; \
	printf "%s\n" "$$output" | grep -q '^Formatted  ' && { echo "Formatting needed. Run: make format"; exit 1; } || exit 0
