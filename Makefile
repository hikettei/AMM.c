CC := gcc

.PHONY: help
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

.PHONY: build_metal
build_metal: ## Build AMM.c w/ Metal
	@echo "Building AMM.c with Metal..."
	@mkdir -p build_metal
	@cmake -S . -B ./build -DCMAKE_C_COMPILER=$(CC) -DAMM_C_METAL=ON _-G Ninja && cd build_metal && ninja

.PHONY: build
build: ## Build AMM.c
	@echo "Building AMM.c..."
	@mkdir -p build
	@cmake -S . -B ./build -DCMAKE_C_COMPILER=$(CC) -G Ninja && cd build && ninja
