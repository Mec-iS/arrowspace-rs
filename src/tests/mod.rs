mod test_arrow;
mod test_builder;
mod test_data;
mod test_helpers;
// mod test_dimensional;
mod test_clustering;
mod test_graph_factory;
mod test_laplacian;
mod test_laplacian_unnormalised;
mod test_querying_proj;
mod test_reduction;
mod test_sparsification;
mod test_taumode;
mod test_eigenmaps;

use std::sync::Once;

static INIT: Once = Once::new();

pub fn init() {
    INIT.call_once(|| {
        // Read RUST_LOG env variable, default to "info" if not set
        let env = env_logger::Env::default().default_filter_or("debug");

        // don't panic if called multiple times across binaries
        let _ = env_logger::Builder::from_env(env)
            .is_test(true) // nicer formatting for tests
            .try_init();
    });
}
