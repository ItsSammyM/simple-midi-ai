use std::collections::HashMap;
use rand::Rng;

use crate::vector::Vector;

const MIN_PITCH: usize = 22;    // A0
const MAX_PITCH: usize = 115;   // Adjusted upper bound
pub const PITCH_RANGE: usize = MAX_PITCH - MIN_PITCH + 1; // Inclusive range
const ASCII_START: u8 = 32;     // First valid ASCII in Cary format ('!') = 33. We use 32 here to also keep track of ' '
const ASCII_END: u8 = 126;      // Last valid ASCII ('~')

pub struct OneHotConverter {
    cache: HashMap<char, Vector>
}

impl OneHotConverter {
    pub fn new() -> Self {
        Self { cache: HashMap::new() }
    }

    pub fn string_to_one_hot<'a>(&'a mut self, string: &'a str) -> impl Iterator<Item = Vector> + 'a {
        string.chars().filter_map(move |c| self.char_to_one_hot(c).ok())
    }

    pub fn char_to_one_hot(&mut self, char: char) -> Result<Vector, &'static str> {
        if let Some(out) = self.cache.get(&char) {
            Ok(out.clone())
        } else {
            let out = Self::char_to_one_hot_calculate(char)?;
            self.cache.insert(char, out.clone());
            Ok(out)
        }
    }
    fn char_to_one_hot_calculate(c: char) -> Result<Vector, &'static str> {
        let ascii_val = c as u8;
        if !(ASCII_START..=ASCII_END).contains(&ascii_val) {
            return Err("Character outside Cary format range (32-126)");
        }
        
        let mut one_hot = Vector::from_zeros(PITCH_RANGE as u8);
        let index = (ascii_val - ASCII_START) as usize;
        one_hot.set(index as u8, 1.0);
        Ok(one_hot)
    }
    

    pub fn one_hot_to_char(vector: &Vector) -> char {
        vector.as_slice()[0..PITCH_RANGE]
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| (i as u32 + ASCII_START as u32))
            .and_then(char::from_u32)
            .unwrap_or(' ')
    }



    pub fn one_hot_to_char_with_randomness(vector: &Vector) -> char {
        const OUTPUT_POSSIBILITIES_COUNT: usize = PITCH_RANGE-1;

        let slice = &vector.as_slice()[0..PITCH_RANGE];
        
        // Early return for empty vectors
        if slice.is_empty() {
            return ' ';
        }
    
        // Get indices sorted by value (descending)
        let mut sorted_indices: Vec<usize> = (0..slice.len()).collect();
        sorted_indices.sort_by(|&a, &b| {
            slice[b].partial_cmp(&slice[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        //top three added
        let sum = sorted_indices
            .iter()
            .take(OUTPUT_POSSIBILITIES_COUNT)
            .map(|i|slice[*i])
            .fold(0.0f32, |f, i|f+i);

        let sorted_indices_ref = &sorted_indices;
        let get_confidence_value = move |index: usize|->f32{
            *slice.get(*sorted_indices_ref.get(index).unwrap_or(&0)).unwrap_or(&1.0)
        };
        
        let random_number: f32 = rand::rng().random_range(0.0..1.0);
        let mut choice = 0;
        let mut boundary = 0.0;
        for i in 0..slice.len().min(OUTPUT_POSSIBILITIES_COUNT){
            boundary = (get_confidence_value(i)/sum)+boundary;
            if random_number < boundary {
                choice = i;
                break;
            }
        }
         
        
        // Clamp choice to valid range
        let clamped_choice = choice.min(sorted_indices.len() - 1);
        
        // Safe conversion to char
        (sorted_indices[clamped_choice] as u32 + ASCII_START as u32)
            .try_into()
            .unwrap_or(' ')
    }
}