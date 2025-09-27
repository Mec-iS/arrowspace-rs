# ArrowSpace

[![DOI](https://joss.theoj.org/papers/10.21105/joss.09002/status.svg)](https://doi.org/10.21105/joss.09002)

**Fast spectral vector search that finds similarity beyond traditional distance metrics**

ArrowSpace is a high-performance Rust library for vector similarity search that goes beyond 
geometric distance (cosine, L2) by incorporating spectral graph properties, enabling more 
nuanced similarity matching for scientific and structured data.

`ArrowSpace` is a data structure library that encapsulates the use of `λτ` indexing; a novel scoring method that mixes Rayleigh and Laplacian scoring (see [`RESEARCH.md`](./RESEARCH.md)) for building vector-search-friendly lookup tables with built-in spectral-awareness. This allows better managing of datasets where spectral characteristics are most relevant. It pairs dense, row‑major arrays with per‑row spectral scores (`λτ`) derived from a Rayleigh-Laplacian score built over items, enabling lambda‑aware similarity, range queries, and composable operations like superposition and element‑wise multiplication over rows. It has been designed to work on datasets where spectral characteristics can be leveraged to find matches that are usually ranked lower by commonly used distance metrics.

Run `cargo run --example proteins_lookup` for an example about how it compares with cosine similarity.

## Usage
```rust
use arrowspace::builder::ArrowSpaceBuilder;

// Simple example that works immediately
let vectors = vec![
    vec![1.0, 2.0, 3.0],
    vec![2.0, 3.0, 1.0], 
    vec![3.0, 1.0, 2.0],
];

let (aspace, _) = ArrowSpaceBuilder::new()
    .with_rows(vectors)
    .build();
```

### Requirements

- Rust 1.78+ (edition 2024)

## Installation

### As a Library Dependency
Add to your `Cargo.toml`:
```toml
[dependencies]
arrowspace = "*"
```

### From Source
```bash
git clone https://github.com/Mec-iS/arrowspace-rs
cd arrowspace-rs
cargo build --release
```

### Running Examples
```bash
cargo run --example compare_cosine
cargo run --example proteins_lookup
```

### Running Tests
```bash
cargo test
```

### Run Bench
```
$ cargo bench
```

### Minimal usage

Construct an `ArrowSpace` from rows and compute a synthetic index `λτ` used in similarities search (spectral search):

- Build λτ‑graph from data (preferred path):
    - Use `ArrowSpaceBuilder::new().build(items)` to get an `ArrowSpace` and its Laplacian+Tau mode; the builder will compute per‑row synthetic indices immediately.
    - Use `ArrowSpaceBuilder::new().with_lambda_graph(...).build(items)` to get an `ArrowSpace` and its Laplacian+Tau mode by specifying the parameters for the graph where the Laplacian is computed.
    - Use `ArrowSpaceBuilder::new().with_lambdas(...).with_synthesis(...).build(items)` to get an `ArrowSpace` and its Laplacian+Tau indices by specifying which lambdas values to use.
- Search the space:
```rust
use arrowSpace::builder::ArrowSpaceBuilder;
use arrowSpace::core::ArrowItem;

// define the search parameters: alpha=1.0 is equivalent to cosine similarity
let alpha = 0.7;
let beta = 0.3;

// Build ArrowSpace from item vectors
let items = vec![
    vec![1.0, 2.0, 3.0],  // Item 1
    vec![2.0, 3.0, 1.0],  // Item 2
    vec![3.0, 1.0, 2.0],  // Item 3
];

let (aspace, _graph) = ArrowSpaceBuilder::new()
    .with_rows(items)
    .with_lambda_graph(0.5, 3, 2.0, sigma: 0.25)
    .build();

// prepare query vector
let query = ArrowItem::new(vec![1.5, 2.5, 2.0], 0.0);
// search the space
let results = aspace.search_lambda_aware(&query, 5, alpha, beta);
println!("{:?}", results);

```

## Main Features (spectral graph construction and search)

- Data structure for vector search:
    - Lambda+Tau graph from data (default): builds a Laplacian over items from the row matrix, then computes per‑row synthetic λτ using laplacian + TauMode (see paper) with Median policy by default; override via `with_synthesis(alpha, mode)` to change α or τ policy.
    - Direct lambda ε‑graph (lower‑level): constructs a Laplacian from a vector of λ values with ε thresholding and k‑capping, union‑symmetrized CSR; use when supplying external λ instead of synthetic.
    - (optional) Hypergraph overlays: build Laplacians from hyperedges (clique expansion, normalized variant) and overlay “boosts” to strengthen pairs; for prebuilt/hypergraph paths, synthetic λ is opt‑in via with_synthesis.
    - (optional) Ensembles: parameterized variants (k adjust, radius/ε expand, hypergraph transforms) for graph experimentation while reusing the same data matrix; synthetic λ is computed per chosen base when enabled.
- Examples:
    - End‑to‑end examples: protein‑like lookup with λ‑band range query using a ZSET‑style index; showcases for hypergraph, λ‑graph, and synthetic laplacian + TauMode flows.
    - Extensive tests spanning `ArrowSpace` algebra, Rayleigh properties, lambda scale‑invariance, superposition bounds, λ‑graph symmetry and k‑capping semantics, hypergraph correctness, diffusion/random‑walk simulations, fractal integrations, and synthetic λ via Median/Mean/Percentile τ policies.

## Key concepts

See [paper](./paper.md)
