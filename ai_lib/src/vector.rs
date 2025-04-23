use std::fmt::Display;

use rand::{rngs::ThreadRng, Rng};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone)]
pub struct Vector(Box<[f32]>);
impl Vector{
    pub fn from_slice(inner: Box<[f32]>)->Self{
        Self(inner)
    }
    pub fn from_zeros(size: u8)->Self{
        Self::from_slice((0..size).map(|_|0.0).collect())
    }
    pub fn new_random(rng: &mut ThreadRng, size: u8, max: f32)->Self{
        (0..size)
            .map(|_|rng.random_range(-max..max))
            .collect::<Box<[f32]>>()
            .into()
    }

    pub fn as_slice(&self)->&Box<[f32]>{
        &self.0
    }
    pub fn as_slice_mut(&mut self)->&mut Box<[f32]>{
        &mut self.0
    }

    pub fn set(&mut self, index: u8, val: f32){
        self.0[index as usize] = val;
    }
    pub fn get(&self, index: u8)->Option<&f32>{
        self.0.get::<usize>(index.into())
    }

    /// If the vectors are of different size, "0"s are added to the end of the smaller one, then the dot product is taken
    pub fn dot(a: &Vector, b: &Vector)->f32{
        a.0.iter().zip(b.0.iter()).fold(0.0, |sum,(a,b)|sum+(a*b))
    }

    pub fn concatenate(a: &Vector, b: &Vector)->Vector{
        Self::from_slice(a.0.iter().chain(b.0.iter()).map(|n|*n).collect())
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