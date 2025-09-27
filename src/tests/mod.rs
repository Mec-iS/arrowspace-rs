#[cfg(test)]
mod test_arrow;
#[cfg(test)]
mod test_builder;
#[cfg(test)]
mod test_graph_factory;
#[cfg(test)]
mod test_laplacian;
#[cfg(test)]
mod test_data;
#[cfg(test)]
mod test_taumode;


use crate::graph::GraphParams;
use crate::taumode::TauMode;

pub const GRAPH_PARAMS: GraphParams =
    GraphParams { eps: 0.1, k: 3, p: 2.0, sigma: None, normalise: true };

pub const TAU_PARAMS: TauMode = TauMode::Median;

