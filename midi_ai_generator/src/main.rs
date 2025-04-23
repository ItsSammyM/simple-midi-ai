use std::{fmt::Write, fs, path::Path};

use ai_lib::{network::{create_network, Network, ONE_HOT_VEC_SIZE}, one_hot_converter::OneHotConverter, vector::Vector};

fn main() {
    let mut converter = OneHotConverter::new();
    let net = Network::load().unwrap_or(create_network());

    let mut out: String = String::new();
    let mut hidden_state = Vector::from_zeros(ONE_HOT_VEC_SIZE);
    let mut output = Vector::from_zeros(ONE_HOT_VEC_SIZE);

    for _ in 0..1000 {
        (output, hidden_state) = net.forward(Vector::concatenate(&output, &hidden_state));
        let output_char = OneHotConverter::one_hot_to_char_with_randomness(&output);
        out.write_char(output_char).unwrap();
        output = converter.char_to_one_hot(output_char).unwrap();
    }


    fs::write(Path::new("../data/output/cary/first.cary"), out).unwrap()
}

