use std::fs::{File, read_dir};
use std::io::{BufReader, BufRead, Write};

const FILE_FOLDER: &str = "../data/input/midicsv/";
const OUTPUT_FOLDER: &str = "../data/input/cary/";


const PITCH_COUNT: usize = 110;
const MINIMUM_PITCH: i32 = 22;

struct MidiProcessor {
    notes: Vec<Vec<i32>>,
    quantization_size: f32,
    allow: [bool; 128],
}

impl MidiProcessor {
    fn new() -> Self {
        MidiProcessor {
            notes: vec![vec![0; PITCH_COUNT]; 150000],
            quantization_size: 40.0,
            allow: [true; 128],
        }
    }

    fn process_file(&mut self, filename: &str) {
        self.reset_state();
        
        let path = format!("{}{}", FILE_FOLDER, filename);
        let file = File::open(&path).expect("Unable to open file");
        let reader = BufReader::new(file);

        for line in reader.lines() {
            let line = line.unwrap();
            let parts: Vec<&str> = line.split(", ").collect();
            
            if parts.len() >= 6 && parts[2] == "Tempo" {
                match (parts[3].parse::<f32>(), parts[5].parse::<f32>()) {
                    (Ok(tempo), Ok(division)) => {
                        self.quantization_size = (50000.0 / tempo) * division;
                    }
                    _ => continue,
                }
            }

            if parts.len() >= 5 && parts[2] == "Program_c" {
                let current_instrument: i32 = parts[4].parse().unwrap();
                let channel: usize = parts[3].parse().unwrap();
                self.allow[channel] = current_instrument >= 0 && current_instrument <= 7;
            }

            if parts.len() >= 6 && !line.contains('"') {
                let track: i32 = parts[0].parse().unwrap();
                let channel: usize = parts[3].parse().unwrap();
                
                if self.allow[channel] && track <= 8 {
                    let event_type = parts[2];
                    let time = (parts[1].parse::<f32>().unwrap() / self.quantization_size) as usize;
                    let pitch: usize = parts[4].parse().unwrap();
                    let volume: i32 = parts[5].parse().unwrap();

                    if time < 150000 {
                        if event_type == "Note_on_c" && volume >= 1 && self.notes[time][pitch] == 0 {
                            self.notes[time][pitch] = 1;
                        } else if (event_type == "Note_on_c" && volume == 0) || event_type == "Note_off_c" {
                            let mut j = time as i32 - 1;
                            while j >= 0 && self.notes[j as usize][pitch] % 2 == 0 {
                                j -= 1;
                            }
                            if j >= 0 {
                                let end = time.max((j + 1) as usize);
                                for k in j as usize..end {
                                    self.notes[k][pitch] = (self.notes[k][pitch] / 2) * 2 + 2;
                                }
                            }
                        }
                    }
                }
            }
        }

        for transposition in -6..6 {
            let output_path = format!("{}/text{}_{}.cary", OUTPUT_FOLDER, filename, transposition);
            let mut output = File::create(output_path).expect("Unable to create file");
            
            for x in 0..150000.min(self.notes.len()) {
                let mut line = String::new();
                for y in 24..PITCH_COUNT {
                    if self.notes[x][y] >= 1 {
                        let the_num = 33 + (y as i32 - MINIMUM_PITCH + transposition);
                        if (33..=126).contains(&the_num) {
                            line.push(the_num as u8 as char);
                        }
                    }
                }
                if !line.is_empty() {
                    line.push(' ');
                }
                if x % 50 == 49 {
                    line.push('\n');
                }
                write!(output, "{}", line).unwrap();
            }
        }
    }

    fn reset_state(&mut self) {
        self.allow = [true; 128];
        for i in 0..150000 {
            for j in 0..PITCH_COUNT {
                self.notes[i][j] = 0;
            }
        }
    }
}

fn main() {
    let mut processor = MidiProcessor::new();
    let files = read_dir(FILE_FOLDER).expect("Unable to read directory");
    
    for entry in files {
        let entry = entry.expect("Unable to read directory entry");
        let filename = entry.file_name().into_string().unwrap();
        
        println!("Processing {}", filename);
        processor.process_file(&filename);
        println!("Completed {}", filename);
    }
}