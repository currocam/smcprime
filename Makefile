.PHONY: dev build test bench profile

dev:
	# 	uvx maturin develop --release --uv doesn't seem to update .so files in my machine
	uv pip install -e .

build:
	uv build

test:
	cargo test
	uv run --with pytest,joblib,tqdm pytest tests/ -v

profile:
	CARGO_PROFILE_RELEASE_DEBUG=2 cargo build --release --bin bench
	samply record ./target/release/bench
