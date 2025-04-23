use std::fs::{self, read_dir};
use ai_lib::{
    network::{create_network, Network, ONE_HOT_VEC_SIZE}, one_hot_converter::OneHotConverter, vector::Vector
};


const INPUT_DIR: &str = "../data/input/cary/";


fn main() {
    let converter = &mut OneHotConverter::new();
    let learning_rate = 10.0;
    let mut net = Network::load().unwrap_or(create_network());

    let all_files_in_dir = read_dir(INPUT_DIR).unwrap();

    for path in all_files_in_dir {
        let Ok(entry) = path else {continue;};
        let path = entry.path();

        println!("File: {}", path.to_str().unwrap());
        let batches = batchify(
            converter,
            fs::read_to_string(path).unwrap()
        );
        train_network(&mut net, &batches, learning_rate);
    }
}


fn train_network(net: &mut Network, batches: &Vec<Vec<Vector>>, learning_rate: f32) {
    for (batch_idx, batch) in batches.iter().enumerate() {
        // Perform backpropagation
        train_from_loss(net, batch, learning_rate);

        if batch_idx % 1000 == 0 {
            net.save();

            // Forward pass to calculate loss
            let mut total_loss = 0.0;
            let mut hidden_state = Vector::from_zeros(ONE_HOT_VEC_SIZE);
            
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
        }
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
    let mut hidden_state = Vector::from_zeros(ONE_HOT_VEC_SIZE);

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
                    weight_gradients: vec![0.0; layer.nodes[0].input_weights.as_slice().len()],
                    bias_gradient: 0.0,
                })
                .collect()
        })
        .collect();

    // We'll do BPTT with a truncated window (simplified)
    const TRUNCATE_STEPS: usize = 20; // How many steps back we propagate
    let seq_len = batch.len();

    for t in (0..seq_len).rev() {
        let (input, output) = &all_activations[t];
        let target = &batch[t];
        
        // Calculate output error
        let error = output.as_slice().iter()
            .zip(target.as_slice().iter())
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
                let output = output.as_slice()[node_idx];
                let derivative = output * (1.0 - output); // Sigmoid derivative
                
                // Error term depends on layer position
                let error_term = if layer_idx == net.layers.len() - 1 {
                    // Output layer
                    error[node_idx] * derivative
                } else {
                    // Hidden layer - sum of contributions to next layer's errors
                    let mut sum = 0.0;
                    for next_node in &net.layers[layer_idx + 1].nodes {
                        let weight = next_node.input_weights.as_slice()[node_idx];
                        sum += weight * derivative;
                    }
                    sum
                };

                // Update weight gradients
                for (weight_idx, input_val) in layer_input.as_slice().iter().enumerate() {
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
            for (weight_idx, weight) in node.input_weights.as_slice_mut().iter_mut().enumerate() {
                *weight -= learning_rate * grad.weight_gradients[weight_idx] / batch.len() as f32;
            }
            
            // Update bias
            node.input_bias -= learning_rate * grad.bias_gradient / batch.len() as f32;
        }
    }
}

fn calculate_loss_of_one_iteration(predicted: &Vector, real: &Vector)->f32{
    real.as_slice().iter().zip(predicted.as_slice().iter())
        .fold(0.0, |fold, (predicted, real)|{
            fold + (*predicted - *real).powi(2)
        })
}

fn batchify(converter: &mut OneHotConverter, string: String) -> Vec<Vec<Vector>> {
    let one_hot_sequence: Vec<_> = converter.string_to_one_hot(&string).collect();
    let sequence_length = one_hot_sequence.len();
    
    let mut batches = Vec::new();
    let mut start = 0;

    const WINDOW_SIZE: usize = 50;
    const MIN_WINDOW_SIZE: usize = 50;
    const WINDOW_STEP: usize = 10;
    while start + WINDOW_SIZE <= sequence_length {
        let end = start + WINDOW_SIZE;
        batches.push(one_hot_sequence[start..end].to_vec());
        start += WINDOW_STEP;
    }

    // Handle remaining elements with padding
    if sequence_length > start + MIN_WINDOW_SIZE {
        let mut final_batch = one_hot_sequence[start..].to_vec();
        while final_batch.len() < WINDOW_SIZE {
            final_batch.push(Vector::from_zeros(ONE_HOT_VEC_SIZE));
        }
        batches.push(final_batch);
    }

    batches
}