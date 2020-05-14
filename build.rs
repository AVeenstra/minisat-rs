extern crate bindgen;
extern crate cc;

use std::env;
use std::path::PathBuf;

pub fn main() {
    let mut builder = cc::Build::new();
    builder.cpp(true)
        .include("lib")
        .include("/usr/include/")
        .define("__STDC_LIMIT_MACROS", None)
        .define("__STDC_FORMAT_MACROS", None);

    if cfg!(feature = "glucose") {
        builder
            .include("lib/glucose-syrup-4.1")
            .file("lib/glucose-syrup-4.1/core/Solver.cc")
            .file("lib/glucose-syrup-4.1/simp/SimpSolver.cc")
            .file("lib/glucose-syrup-4.1/utils/System.cc")
            .file("lib/minisat-c-bindings/minisat.cc")
            .flag("-std=c++11")
            .define("USE_GLUCOSE", None)
            .compile("minisat");
    } else {
        builder
            .include("lib/minisat")
            .file("lib/minisat/minisat/core/Solver.cc")
            .file("lib/minisat/minisat/simp/SimpSolver.cc")
            .file("lib/minisat/minisat/utils/System.cc")
            .file("lib/minisat-c-bindings/minisat.cc")
            .compile("minisat");
    }

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindgen::Builder::default()
        .header("lib/minisat-c-bindings/minisat.h")
        .generate()
        .expect("Could not create bindings to library")
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}
