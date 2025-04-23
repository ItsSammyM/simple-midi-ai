use std::{f32::consts::E, fs, path::Path};

use rand::{rngs::ThreadRng, Rng};
use serde::{Deserialize, Serialize};

use crate::{one_hot_converter::PITCH_RANGE, vector::Vector};

pub const ONE_HOT_VEC_SIZE: u8 = PITCH_RANGE as u8;
pub const SAVE_NET_PATH: &str = "../checkpoints/saved_net_2.txt";


pub fn create_network()->Network{
    let mut rng = rand::rng();

    Network::new_random(
        &mut rng,
        &[ONE_HOT_VEC_SIZE*2, ONE_HOT_VEC_SIZE, ONE_HOT_VEC_SIZE, ONE_HOT_VEC_SIZE]
    )
}




#[derive(Serialize, Deserialize)]
pub struct Network{
    pub layers: Box<[Layer]>
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

    pub fn forward(&self, input: Vector)->(Vector,Vector){
        
        self.layers
            .iter()
            .enumerate()
            .fold((input, Vector::from_zeros(0)), |(data_vec, second_to_last), (idx, layer)|{
                (
                    layer.forward(&data_vec),
                    if idx == self.layers.len() {data_vec} else {second_to_last}
                )
            })
    }


    pub fn save(&self){
        let Ok(string) = serde_json::to_string(self) else {println!("Failed to save"); return;};
        let Ok(_) = fs::write(Path::new(SAVE_NET_PATH), string) else {println!("Failed to save"); return;};
    }
    pub fn load()->Option<Self>{
        let Ok(string) = fs::read_to_string(Path::new(SAVE_NET_PATH)) else {println!("Failed to Load"); return None};
        let Ok(net) = serde_json::from_str::<Network>(&string) else {println!("Failed to Load"); return None};
        return Some(net);
    }

}

#[derive(Serialize, Deserialize)]
pub struct Layer{
    pub nodes: Box<[Node]>
}
impl Layer{
    fn new_random(rng: &mut ThreadRng, previous_layer_size: u8, layer_size: u8)->Self{
        Self{
            nodes: (0..layer_size).map(|_|Node::new_random(rng, previous_layer_size)).collect()
        }
    }

    /// Output vec size = number of nodes
    pub fn forward(&self, input: &Vector)->Vector{
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
pub struct Node{
    pub input_bias: f32,
    pub input_weights: Vector
}
impl Node{
    fn new_random(rng: &mut ThreadRng, previous_layer_size: u8)->Self{
        Self{
            input_bias: rng.random_range(-Network::INITIAL_WEIGHT_MAX..Network::INITIAL_WEIGHT_MAX),
            input_weights: Vector::new_random(rng, previous_layer_size, Network::INITIAL_WEIGHT_MAX)
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
