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

use std::{collections::HashMap, f32::consts::E, fmt::Display, fs, path::Path};
use rand::{self, rngs::ThreadRng, Rng};
use serde::{Deserialize, Serialize};

const ONE_HOT_VEC_SIZE: u8 = 111;

fn main() {
    let mut net = load_net().unwrap_or(create_network());

    let batches = batchify(
        &mut CharToOneHot::new(),
        fs::read_to_string(Path::new("../data/input/cary/t808.csv_0.cary")).unwrap()
    );
    let learning_rate = 0.01;
    
    // Training loop
    for epoch in 0..10 {
        println!("Epoch {}", epoch);
        train_network(&mut net, &batches, learning_rate);
        
        // Calculate validation loss if you have validation data
        let val_loss = calculate_loss_of_batch(&net, &batches[0]);
        println!("Epoch {} - Validation Loss: {:.6}", epoch, val_loss);
        save_net(&net);
    }
}

fn save_net(net: &Network){
    let Ok(string) = serde_json::to_string(net) else {println!("Failed to save"); return;};
    let Ok(_) = fs::write(Path::new("../checkpoints/saved_net"), string) else {println!("Failed to save"); return;};
}
fn load_net()->Option<Network>{
    let Ok(string) = fs::read_to_string(Path::new("../checkpoints/saved_net")) else {println!("Failed to Load"); return None};
    let Ok(net) = serde_json::from_str::<Network>(&string) else {println!("Failed to Load"); return None};
    return Some(net);
}



fn create_network()->Network{
    let mut rng = rand::rng();

    Network::new_random(
        &mut rng,
        &[ONE_HOT_VEC_SIZE*2, ONE_HOT_VEC_SIZE, ONE_HOT_VEC_SIZE]
    )
}

fn train_network(net: &mut Network, batches: &Vec<Vec<Vector>>, learning_rate: f32) {
    for (batch_idx, batch) in batches.iter().enumerate() {
        // Forward pass to calculate loss
        let mut total_loss = 0.0;
        let mut hidden_state = Vector::zeros(ONE_HOT_VEC_SIZE);
        
        for char in batch {
            let input = Vector::concatenate(char, &hidden_state);
            let (output, new_hidden) = net.forward(input);
            hidden_state = new_hidden;
            
            // Calculate and accumulate loss for this time step
            total_loss += calculate_loss_of_one_iteration(&output, char);
        }
        
        // Print loss before backpropagation
        let avg_loss = total_loss / batch.len() as f32;
        println!("Batch {} - Loss: {:.6}", batch_idx, avg_loss);
        
        // Perform backpropagation
        train_from_loss(net, batch, learning_rate);
    }
}


fn train_from_loss(net: &mut Network, batch: &Vec<Vector>, learning_rate: f32) {
    struct NodeGradient {
        weight_gradients: Vec<f32>,
        bias_gradient: f32,
    }

    // Forward pass: store all activations for BPTT
    let mut all_activations = Vec::new();
    let mut all_hidden_states = Vec::new();
    let mut hidden_state = Vector::zeros(ONE_HOT_VEC_SIZE);

    for char in batch {
        let input = Vector::concatenate(char, &hidden_state);
        let (output, new_hidden) = net.forward(input.clone());
        
        all_activations.push((input, output.clone()));
        all_hidden_states.push(hidden_state.clone());
        hidden_state = new_hidden;
    }

    // Backward pass (BPTT)
    let mut gradients: Vec<Vec<NodeGradient>> = net.layers.iter()
        .map(|layer| {
            layer.nodes.iter()
                .map(|_| NodeGradient {
                    weight_gradients: vec![0.0; layer.nodes[0].input_weights.0.len()],
                    bias_gradient: 0.0,
                })
                .collect()
        })
        .collect();

    // We'll do BPTT with a truncated window (simplified)
    const TRUNCATE_STEPS: usize = 5; // How many steps back we propagate
    let seq_len = batch.len();

    for t in (0..seq_len).rev() {
        let (input, output) = &all_activations[t];
        let target = &batch[t];
        
        // Calculate output error
        let error = output.0.iter()
            .zip(target.0.iter())
            .map(|(o, t)| o - t)
            .collect::<Vec<f32>>();

        // Backpropagate through layers
        for layer_idx in (0..net.layers.len()).rev() {
            let layer = &net.layers[layer_idx];
            let layer_input = if layer_idx == 0 {
                input.clone()
            } else {
                // For hidden layers, we need to get the input from the previous layer's output
                // This is simplified - in a full implementation we'd track all layer activations
                net.layers[0..layer_idx].iter()
                    .fold(input.clone(), |acc, l| l.forward(&acc))
            };

            for (node_idx, _) in layer.nodes.iter().enumerate() {
                // Compute gradient for this node
                let output = output.0[node_idx];
                let derivative = output * (1.0 - output); // Sigmoid derivative
                
                // Error term depends on layer position
                let error_term = if layer_idx == net.layers.len() - 1 {
                    // Output layer
                    error[node_idx] * derivative
                } else {
                    // Hidden layer - sum of contributions to next layer's errors
                    let mut sum = 0.0;
                    for next_node in &net.layers[layer_idx + 1].nodes {
                        let weight = next_node.input_weights.0[node_idx];
                        sum += weight * derivative;
                    }
                    sum
                };

                // Update weight gradients
                for (weight_idx, input_val) in layer_input.0.iter().enumerate() {
                    gradients[layer_idx][node_idx].weight_gradients[weight_idx] += 
                        error_term * input_val;
                }

                // Update bias gradient
                gradients[layer_idx][node_idx].bias_gradient += error_term;
            }
        }

        // Stop backpropagating if we've gone far enough back in time
        if seq_len - t > TRUNCATE_STEPS {
            break;
        }
    }

    // Apply gradients
    for (layer_idx, layer) in net.layers.iter_mut().enumerate() {
        for (node_idx, node) in layer.nodes.iter_mut().enumerate() {
            let grad = &gradients[layer_idx][node_idx];
            
            // Update weights
            for (weight_idx, weight) in node.input_weights.0.iter_mut().enumerate() {
                *weight -= learning_rate * grad.weight_gradients[weight_idx] / batch.len() as f32;
            }
            
            // Update bias
            node.input_bias -= learning_rate * grad.bias_gradient / batch.len() as f32;
        }
    }
}



fn calculate_loss_of_batch(net: &Network, batch: &Vec<Vector>)->f32{
    let mut total_loss = 0.0;
    let mut previous = Vector::zeros(ONE_HOT_VEC_SIZE);
    for char in batch{
        let (out, inner) = net.forward(Vector::concatenate(char, &previous));
        previous = inner;

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
    fn one_hot_to_char_calculate(vector: Vector)->Option<char>{
        let mut max = (0, vector.get(0));
        for slot in vector.inner() {
            if slot > max.1 {
                max = (slot, max)
            }
        }
        Some(max)
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

    fn forward(&self, input: Vector)->(Vector,Vector){
        
        self.layers
            .iter()
            .enumerate()
            .fold((input, Vector::zeros(0)), |(data_vec, second_to_last), (idx, layer)|{
                (
                    layer.forward(&data_vec),
                    if idx == self.layers.len() {data_vec} else {second_to_last}
                )
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