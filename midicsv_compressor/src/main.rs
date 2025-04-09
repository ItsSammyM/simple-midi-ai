use std::fs::{File, read_dir};
use std::io::{BufReader, BufRead, Write};
use std::path::Path;

// Constants
const INPUT_DIR: &str = "../data/input/midicsv/";
const OUTPUT_DIR: &str = "../data/input/cary/";
const MAX_PITCHES: usize = 110;
const MIN_PITCH: i32 = 22;
const MAX_TIME_STEPS: usize = 150_000;

#[derive(Clone, Copy, PartialEq)]
enum NoteState {
    Off,
    On,
    Sustained,
}

struct MidiProcessor {
    note_matrix: Vec<[NoteState; MAX_PITCHES]>,
    time_quantum: f32,
    allowed_channels: [bool; 128],
}

impl MidiProcessor {
    fn new() -> Self {
        MidiProcessor {
            note_matrix: vec![[NoteState::Off; MAX_PITCHES]; MAX_TIME_STEPS],
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

        if time_step >= MAX_TIME_STEPS || pitch >= MAX_PITCHES {
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
            
            let mut output_file = File::create(output_path).expect("Failed to create output file");
            
            for (_, notes) in self.note_matrix.iter().enumerate() {
                let mut output_line = String::new();
                
                // Convert active notes to ASCII characters
                for (pitch, state) in notes.iter().enumerate().skip(24) {
                    if *state != NoteState::Off {
                        let ascii_code = 33 + (pitch as i32 - MIN_PITCH + transposition);
                        if (33..=126).contains(&ascii_code) {
                            output_line.push(ascii_code as u8 as char);
                        }
                    }
                }
                
                // Add formatting
                if !output_line.is_empty() {
                    output_line.push(' ');
                }
                
                write!(output_file, "{}", output_line).unwrap();
            }
        }
    }

    fn reset_state(&mut self) {
        self.allowed_channels = [true; 128];
        self.note_matrix = vec![[NoteState::Off; MAX_PITCHES]; MAX_TIME_STEPS];
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


// use std::fs::{File, read_dir};
// use std::io::{BufReader, BufRead, Write};

// const FILE_FOLDER: &str = "../data/input/midicsv/";
// const OUTPUT_FOLDER: &str = "../data/input/cary/";


// const PITCH_COUNT: usize = 110;
// const MINIMUM_PITCH: i32 = 22;

// struct MidiProcessor {
//     notes: Vec<Vec<i32>>,
//     quantization_size: f32,
//     allow: [bool; 128],
// }

// impl MidiProcessor {
//     fn new() -> Self {
//         MidiProcessor {
//             notes: vec![vec![0; PITCH_COUNT]; 150000],
//             quantization_size: 40.0,
//             allow: [true; 128],
//         }
//     }

//     fn process_file(&mut self, filename: &str) {
//         self.reset_state();
        
//         let path = format!("{}{}", FILE_FOLDER, filename);
//         let file = File::open(&path).expect("Unable to open file");
//         let reader = BufReader::new(file);

//         for line in reader.lines() {
//             let line = line.unwrap();
//             let parts: Vec<&str> = line.split(", ").collect();
            
//             if parts.len() >= 6 && parts[2] == "Tempo" {
//                 match (parts[3].parse::<f32>(), parts[5].parse::<f32>()) {
//                     (Ok(tempo), Ok(division)) => {
//                         self.quantization_size = (50000.0 / tempo) * division;
//                     }
//                     _ => continue,
//                 }
//             }

//             if parts.len() >= 5 && parts[2] == "Program_c" {
//                 let current_instrument: i32 = parts[4].parse().unwrap();
//                 let channel: usize = parts[3].parse().unwrap();
//                 self.allow[channel] = current_instrument >= 0 && current_instrument <= 7;
//             }

//             if parts.len() >= 6 && !line.contains('"') {
//                 let track: i32 = parts[0].parse().unwrap();
//                 let channel: usize = parts[3].parse().unwrap();
                
//                 if self.allow[channel] && track <= 8 {
//                     let event_type = parts[2];
//                     let time = (parts[1].parse::<f32>().unwrap() / self.quantization_size) as usize;
//                     let pitch: usize = parts[4].parse().unwrap();
//                     let volume: i32 = parts[5].parse().unwrap();

//                     if time < 150000 {
//                         if event_type == "Note_on_c" && volume >= 1 && self.notes[time][pitch] == 0 {
//                             self.notes[time][pitch] = 1;
//                         } else if (event_type == "Note_on_c" && volume == 0) || event_type == "Note_off_c" {
//                             let mut j = time as i32 - 1;
//                             while j >= 0 && self.notes[j as usize][pitch] % 2 == 0 {
//                                 j -= 1;
//                             }
//                             if j >= 0 {
//                                 let end = time.max((j + 1) as usize);
//                                 for k in j as usize..end {
//                                     self.notes[k][pitch] = (self.notes[k][pitch] / 2) * 2 + 2;
//                                 }
//                             }
//                         }
//                     }
//                 }
//             }
//         }

//         for transposition in -6..6 {
//             let output_path = format!("{}/text{}_{}.cary", OUTPUT_FOLDER, filename, transposition);
//             let mut output = File::create(output_path).expect("Unable to create file");
            
//             for x in 0..150000.min(self.notes.len()) {
//                 let mut line = String::new();
//                 for y in 24..PITCH_COUNT {
//                     if self.notes[x][y] >= 1 {
//                         let the_num = 33 + (y as i32 - MINIMUM_PITCH + transposition);
//                         if (33..=126).contains(&the_num) {
//                             line.push(the_num as u8 as char);
//                         }
//                     }
//                 }
//                 if !line.is_empty() {
//                     line.push(' ');
//                 }
//                 if x % 50 == 49 {
//                     line.push('\n');
//                 }
//                 write!(output, "{}", line).unwrap();
//             }
//         }
//     }

//     fn reset_state(&mut self) {
//         self.allow = [true; 128];
//         for i in 0..150000 {
//             for j in 0..PITCH_COUNT {
//                 self.notes[i][j] = 0;
//             }
//         }
//     }
// }

// fn main() {
//     let mut processor = MidiProcessor::new();
//     let files = read_dir(FILE_FOLDER).expect("Unable to read directory");
    
//     for entry in files {
//         let entry = entry.expect("Unable to read directory entry");
//         let filename = entry.file_name().into_string().unwrap();
        
//         println!("Processing {}", filename);
//         processor.process_file(&filename);
//         println!("Completed {}", filename);
//     }
// }