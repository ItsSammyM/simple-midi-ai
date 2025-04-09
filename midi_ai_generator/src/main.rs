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

use std::{collections::HashMap, f32::consts::E, fmt::Display, fs::{self, File}, io::{BufRead, BufReader, Read}, ops::Deref, path::{self, Path}};
use rand::{self, rngs::ThreadRng, Rng};
use serde::{Deserialize, Serialize};

const ONE_HOT_VEC_SIZE: u8 = 111;


fn main() {
    println!("Hello, world!");

    // let serialized = serde_json::to_string(&neural_net).unwrap();
    // let neural_net: Network = serde_json::from_str(serialized.as_str()).unwrap();

    let mut net = create_network();
    println!("reading batches");
    let batches = batchify(
        &mut CharToOneHot::new(),
        fs::read_to_string(Path::new("../data/input/cary/t808.csv_0.cary")).unwrap()
    );
    println!("training net");
    train_network(&mut net, &batches);
    
    // println!("{}", output);
}



fn create_network()->Network{
    let mut rng = rand::rng();

    Network::new_random(
        &mut rng,
        &[ONE_HOT_VEC_SIZE*2, ONE_HOT_VEC_SIZE, ONE_HOT_VEC_SIZE]
    )
}

fn train_network(net: &mut Network, batches: &Vec<Vec<Vector>>){
    for (i, batch) in batches.iter().enumerate(){
        let loss = calculate_loss_of_batch(net, batch);
        train_from_loss(net, loss);
        println!("(Batch, Loss): ({i}, {loss})",)
    }
}


fn train_from_loss(net: &mut Network, loss: f32){
    //backprop the entire net
}

fn calculate_loss_of_batch(net: &mut Network, batch: &Vec<Vector>)->f32{
    let mut total_loss = 0.0;
    let previous = Vector::zeros(ONE_HOT_VEC_SIZE);
    for char in batch{
        let out = net.forward(Vector::concatenate(char, &previous));

        total_loss += calculate_loss_of_one_iteration(&out, char)
    }
    total_loss
}
fn calculate_loss_of_one_iteration(predicted: &Vector, real: &Vector)->f32{
    real.0.iter().zip(predicted.0.iter())
        .fold(0.0, |fold, (predicted, real)|{
            fold + (*predicted - *real).powi(2)
        })
}


fn batchify(converter: &mut CharToOneHot, string: String) -> Vec<Vec<Vector>> {
    let one_hot_sequence: Vec<_> = converter.string_to_one_hot(&string).collect();
    let sequence_length = one_hot_sequence.len();
    
    let mut batches = Vec::new();
    let mut start = 0;

    const WINDOW_SIZE: usize = 100;
    const MIN_WINDOW_SIZE: usize = 100;
    const WINDOW_STEP: usize = 1;
    while start + WINDOW_SIZE <= sequence_length {
        let end = start + WINDOW_SIZE;
        batches.push(one_hot_sequence[start..end].to_vec());
        start += WINDOW_STEP;
    }

    // Handle remaining elements with padding
    if sequence_length > start + MIN_WINDOW_SIZE {
        let mut final_batch = one_hot_sequence[start..].to_vec();
        while final_batch.len() < WINDOW_SIZE {
            final_batch.push(Vector::zeros(ONE_HOT_VEC_SIZE));
        }
        batches.push(final_batch);
    }

    batches
}


struct CharToOneHot{
    cache: HashMap<char, Vector>
}
impl CharToOneHot{
    pub fn string_to_one_hot<'a, 'b>(&'b mut self, string: &'a String)->impl Iterator<Item=Vector> + use<'a, 'b>{
        string.chars().filter_map(|char|self.char_to_one_hot(char).ok())
    }
    fn new()->Self{
        Self{cache: HashMap::new()}
    }
    fn char_to_one_hot(&mut self, char: char)->Result<Vector, &'static str>{
        if let Some(out) = self.cache.get(&char){
            Ok(out.clone())
        }else{
            let out = Self::char_to_one_hot_calculate(char)?;
            self.cache.insert(char, out.clone());
            Ok(out)
        }
    }
    fn char_to_one_hot_calculate(c: char) -> Result<Vector, &'static str> {
        let base = c as u8;
        if !(0..94).contains(&base) {
            return Err("Invalid character for Cary format");
        }
        let mut one_hot = Vector::zeros(ONE_HOT_VEC_SIZE);
        one_hot.set(base, 1.0);
        Ok(one_hot)
    }
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

    fn zeros(size: u8)->Self{
        Self::new((0..size).map(|_|0.0).collect())
    }

    fn set(&mut self, index: u8, val: f32){
        self.0[index as usize] = val;
    }
    fn get(&self, index: u8)->Option<&f32>{
        self.0.get::<usize>(index.into())
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

    fn concatenate(a: &Vector, b: &Vector)->Vector{
        Self::new(a.0.iter().chain(b.0.iter()).map(|n|*n).collect())
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