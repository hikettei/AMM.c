CC := gcc

.PHONY: help
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

.PHONY: build
build: ## Build AMM.c
	@echo "Building AMM.c..."
	@mkdir -p build
	@cmake -S . -B ./build -DCMAKE_C_COMPILER=$(CC) -G Ninja && cd build && ninja

.PHONY: test
test: ## Run unit tests
	@echo "Building AMM.c..."
	@mkdir -p build
	@cmake -S . -B ./build -DCMAKE_C_COMPILER=$(CC) -DAMM_C_BUILD_TESTS=ON -G Ninja && cd build && ninja
	@echo "Running unit tests..."
	@./build/amm_test
