.PHONY: dev build test

dev:
	# 	uvx maturin develop --release --uv doesn't seem to update .so files in my machine
	uv pip install -e .

build:
	uv build

test:
	cargo test
	uv run --with pytest,joblib,tqdm pytest tests/ -v
