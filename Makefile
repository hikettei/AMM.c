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

build_test: ## Build AMM.c with unittests
	@echo "Building AMM.c with tests..."
	@mkdir -p build
	@cmake -S . -B ./build -DCMAKE_C_COMPILER=$(CC) -DAMM_C_SAFE_MODE=ON -DAMM_C_BUILD_TESTS=ON -G Ninja && cd build && ninja

test_ndarray: build_test ## Run ndarray unittests
	@./build/amm_test_ndarray

test_amm1: build_test ## Run AMM1 unittests
	@./build/amm_test_amm1

test: build_test ## Run all unittests
	@./build/amm_test_ndarray
	@./build/amm_test_amm1
