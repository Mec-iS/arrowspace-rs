/// Copyright (2025) Mec-iS, tuned.org.uk
//    Licensed under the Apache License, Version 2.0 (the "License");
//    you may not use this file except in compliance with the License.
//    You may obtain a copy of the License at
//
//        http://www.apache.org/licenses/LICENSE-2.0
//
//    Unless required by applicable law or agreed to in writing, software
//    distributed under the License is distributed on an "AS IS" BASIS,
//    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//    See the License for the specific language governing permissions and
//    limitations under the License.
pub mod builder;
pub mod clustering;
pub mod core;
pub mod graph;
pub mod laplacian;
pub mod reduction;
pub mod sampling;
pub mod sparsification;
pub mod taumode;

#[cfg(test)]
mod tests;

use std::sync::Once;

static INIT: Once = Once::new();

pub fn init() {
    INIT.call_once(|| {
        // Read RUST_LOG env variable, default to "info" if not set
        let env = env_logger::Env::default().default_filter_or("info");

        // don't panic if called multiple times across binaries
        let _ = env_logger::Builder::from_env(env)
            .is_test(true) // nicer formatting for tests
            .try_init();
    });
}
