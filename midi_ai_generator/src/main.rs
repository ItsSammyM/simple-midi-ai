/*
    One Hot Vector Example
    a = [1, 0, 0]
    b = [0, 1, 0]
    c = [0, 0, 1]
    And plausible neural net outputs
    a(with .8 confidence) = [.8, .1, .01]

    Neural net function
    Input -> Output
    OneHotVector-Character -> OneHotVector-Character

    Deepseeks Loss Function
    Cross-entrypy loss
    Loss = -log(predicted_probability_of_correct_character)
    If the correct output is [1, 0, 0] and the model predicts [.8, .1, .01] then loss = -log(.8)

*/

use std::{f32::consts::E, fmt::Display};
use rand::{self, rngs::ThreadRng, Rng};
use serde::{Deserialize, Serialize};

const ONE_HOT_VEC_SIZE: u8 = 110;


fn main() {
    println!("Hello, world!");

    let mut rng = rand::rng();
    let neural_net = Network::new_random(&mut rng, &[5, 5, 5]);

    let input = Vector(Box::new([0.0,0.0,0.0,0.0,0.0]));

    let output = neural_net.forward(input.clone());
    println!("{}", output);

    let serialized = serde_json::to_string(&neural_net).unwrap();
    let neural_net: Network = serde_json::from_str(serialized.as_str()).unwrap();
    
    let output = neural_net.forward(input);
    println!("{}", output);
}

#[derive(Serialize, Deserialize)]
struct Network{
    layers: Box<[Layer]>
}
impl Network{
    const INITIAL_WEIGHT_MAX: f32 = 1.0;


    fn new_random(rng: &mut ThreadRng, layer_sizes: &[u8])->Self{
        let a = layer_sizes.iter();
        let mut b = layer_sizes.iter();
        b.next();

        Self{
            layers: a.zip(b)
                .map(|(first, second)|Layer::new_random(rng, *first, *second))
                .collect()
        }
    }

    fn forward(&self, input: Vector)->Vector{
        self.layers
            .iter()
            .fold(input, |data_vec, layer|{
                layer.forward(&data_vec)
            })
    }
}

#[derive(Serialize, Deserialize)]
struct Layer{
    nodes: Box<[Node]>
}
impl Layer{
    fn new_random(rng: &mut ThreadRng, previous_layer_size: u8, layer_size: u8)->Self{
        Self{
            nodes: (0..layer_size).map(|_|Node::new_random(rng, previous_layer_size)).collect()
        }
    }

    /// Output vec size = number of nodes
    fn forward(&self, input: &Vector)->Vector{
        self.nodes
            .iter()
            .map(|node|
                node.forward(input)
            )
            .collect::<Box<[f32]>>()
            .into()
    }
}

#[derive(Serialize, Deserialize)]
struct Node{
    input_bias: f32,
    input_weights: Vector
}
impl Node{
    fn new_random(rng: &mut ThreadRng, previous_layer_size: u8)->Self{
        Self{
            input_bias: rng.random_range(-Network::INITIAL_WEIGHT_MAX..Network::INITIAL_WEIGHT_MAX),
            input_weights: Vector::new_random(rng, previous_layer_size)
        }
        
    }

    fn forward(&self, input: &Vector)->f32{
        Self::activation(Vector::dot(
            &self.input_weights,
            input
        ) + self.input_bias)
    }

    fn activation(x: f32)->f32{
        1.0 / (1.0 + E.powf(-x))
    }
}

#[derive(Serialize, Deserialize, Clone)]
struct Vector(Box<[f32]>);
impl Vector{
    fn new(inner: Box<[f32]>)->Self{
        Self(inner)
    }

    fn new_random(rng: &mut ThreadRng, size: u8)->Self{
        (0..size)
            .map(|_|rng.random_range(-Network::INITIAL_WEIGHT_MAX..Network::INITIAL_WEIGHT_MAX))
            .collect::<Box<[f32]>>()
            .into()
    }

    /// If the vectors are of different size, "0"s are added to the end of the smaller one, then the dot product is taken
    fn dot(a: &Vector, b: &Vector)->f32{
        a.0.iter().zip(b.0.iter()).fold(0.0, |sum,(a,b)|sum+(a*b))
    }
}
impl From<Box<[f32]>> for Vector{
    fn from(value: Box<[f32]>) -> Self {
        Self(value)
    }
}
impl Display for Vector{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[")?;
        for i in self.0.iter() {
            write!(f, "{}, ", i)?;
        }
        write!(f, "]")?;
        Ok(())
    }
}