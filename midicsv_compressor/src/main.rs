use std::fs::{File, read_dir};
use std::io::{BufReader, BufRead, Write};
use std::path::{Path, PathBuf};

// Constants
const INPUT_DIR: &str = "../data/input/midicsv/";
const OUTPUT_DIR: &str = "../data/input/cary/";


const MIN_PITCH: usize = 22;    // A0
const MAX_PITCH: usize = 110;
const MAX_TIME_STEPS: usize = 150_000;
const ASCII_OFFSET: i32 = 33; // First printable ASCII '!' -- Space is 32

#[derive(Clone, Copy, PartialEq)]
enum NoteState {
    Off,
    On,
    Sustained,
}

struct MidiProcessor {
    note_matrix: Vec<[NoteState; MAX_PITCH]>,
    time_quantum: f32,
    allowed_channels: [bool; 128],
}

impl MidiProcessor {
    fn new() -> Self {
        MidiProcessor {
            note_matrix: vec![[NoteState::Off; MAX_PITCH]; MAX_TIME_STEPS],
            time_quantum: 40.0,
            allowed_channels: [true; 128],
        }
    }

    fn process_file(&mut self, filename: &str) {
        self.reset_state();
        
        let file_path = Path::new(INPUT_DIR).join(filename);
        let file = File::open(&file_path).expect("Failed to open input file");
        let reader = BufReader::new(file);

        for line in reader.lines() {
            let line = line.expect("Failed to read line");
            let parts: Vec<&str> = line.split(", ").collect();
            
            self.process_tempo_change(&parts);
            self.process_instrument_change(&parts);
            self.process_note_event(&parts);
        }

        self.generate_output_files(filename);
    }

    fn process_tempo_change(&mut self, parts: &[&str]) {
        if parts.len() >= 6 && parts[2] == "Tempo" {
            if let (Ok(tempo), Ok(division)) = (parts[3].parse::<f32>(), parts[5].parse::<f32>()) {
                self.time_quantum = (50_000.0 / tempo) * division;
            }
        }
    }

    fn process_instrument_change(&mut self, parts: &[&str]) {
        if parts.len() >= 5 && parts[2] == "Program_c" {
            if let (Ok(channel), Ok(instrument)) = (parts[3].parse::<usize>(), parts[4].parse::<i32>()) {
                // Only allow piano-like instruments (0-7)
                self.allowed_channels[channel] = (0..=7).contains(&instrument);
            }
        }
    }

    fn process_note_event(&mut self, parts: &[&str]) {
        if parts.len() < 6 || parts[2].contains('"') {
            return;
        }

        let track: i32 = parts[0].parse().unwrap();
        let channel: usize = parts[3].parse().unwrap();
        
        if !self.allowed_channels[channel] || track > 8 {
            return;
        }

        let event_type = parts[2];
        let time_step = (parts[1].parse::<f32>().unwrap() / self.time_quantum) as usize;
        let pitch: usize = parts[4].parse().unwrap();
        let velocity: i32 = parts[5].parse().unwrap();

        if time_step >= MAX_TIME_STEPS || pitch >= MAX_PITCH {
            return;
        }

        match (event_type, velocity) {
            ("Note_on_c", v) if v >= 1 => self.handle_note_on(time_step, pitch),
            ("Note_on_c", 0) | ("Note_off_c", _) => self.handle_note_off(time_step, pitch),
            _ => (),
        }
    }

    fn handle_note_on(&mut self, time: usize, pitch: usize) {
        if self.note_matrix[time][pitch] == NoteState::Off {
            self.note_matrix[time][pitch] = NoteState::On;
        }
    }

    fn handle_note_off(&mut self, time: usize, pitch: usize) {
        // Find when the note was last played
        let mut last_on_time = time.saturating_sub(1);
        while last_on_time > 0 && self.note_matrix[last_on_time][pitch] != NoteState::On {
            last_on_time -= 1;
        }

        // Mark all times between last_on and now as sustained
        if self.note_matrix[last_on_time][pitch] == NoteState::On {
            for t in last_on_time..time {
                if self.note_matrix[t][pitch] == NoteState::Off {
                    self.note_matrix[t][pitch] = NoteState::Sustained;
                }
            }
        }
    }

    fn generate_output_files(&self, filename: &str) {
        for transposition in -6..6 {
            let output_path = Path::new(OUTPUT_DIR)
                .join(format!("{}_{}.cary", filename, transposition));
            println!("Attempting generating of {:?}", output_path);
            
            self.generate_output_file(output_path, transposition).expect("Failed to create file");
        }
    }

    

    fn generate_output_file(&self, output_path: PathBuf, transpose: i32) -> std::io::Result<()> {
        let mut output_file = File::create(output_path)?;
        
        for frame in &self.note_matrix {
            let mut output = String::with_capacity(16); // Reduce allocations
            
            for (pitch, state) in frame.iter().enumerate().skip(MIN_PITCH) {
                if *state != NoteState::Off {
                    let adjusted_pitch = pitch as i32 - MIN_PITCH as i32 + transpose;
                    let c = (ASCII_OFFSET + adjusted_pitch) as u8 as char;
                    if c.is_ascii_graphic() {
                        output.push(c);
                    }
                }
            }
            
            if !output.is_empty() {
                output.push(' ');
                output_file.write_all(output.as_bytes())?;
            }
        }
        
        Ok(())
    }

    fn reset_state(&mut self) {
        self.allowed_channels = [true; 128];
        self.note_matrix = vec![[NoteState::Off; MAX_PITCH]; MAX_TIME_STEPS];
    }
}

fn main() {
    let mut processor = MidiProcessor::new();
    let input_dir = read_dir(INPUT_DIR).expect("Failed to read input directory");
    
    for entry in input_dir {
        let entry = entry.expect("Failed to read directory entry");
        let filename = entry.file_name().into_string().unwrap();
        
        println!("Processing {}", filename);
        processor.process_file(&filename);
        println!("Completed {}", filename);
    }
}