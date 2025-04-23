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

pub mod one_hot_converter;
pub mod vector;
pub mod network;



