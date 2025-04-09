use std::fs::{File, read_dir};
use std::io::{BufRead, BufReader, BufWriter, Write};

const INPUT_FOLDER: &str = "../data/input/cary/";
const OUTPUT_FOLDER: &str = "../data/output/midicsv/";


const MULTI: u32 = 40;
const PITCH_COUNT: usize = 116;
const MAX_TIME_STEPS: usize = 20000;

struct MidiDecompressor {
    notes: Vec<Vec<bool>>,
    pointer_at: usize,
}

impl MidiDecompressor {
    fn new() -> Self {
        MidiDecompressor {
            notes: vec![vec![false; PITCH_COUNT]; MAX_TIME_STEPS],
            pointer_at: 0,
        }
    }

    fn load_compressed_file(&mut self, path: &str) -> std::io::Result<()> {
        self.notes.iter_mut().for_each(|row| row.fill(false));
        self.pointer_at = 0;

        let file = File::open(path)?;
        let reader = BufReader::new(file);

        for line in reader.lines() {
            let line = line?;
            for c in line.chars() {
                if c == ' ' {
                    self.pointer_at += 1;
                } else {
                    let pitch = c as i32 - 32 - 1;
                    if pitch >= 0 && (pitch as usize) < PITCH_COUNT {
                        self.notes[self.pointer_at][pitch as usize] = true;
                    }
                }
            }
        }
        self.pointer_at += 1;
        Ok(())
    }

    fn generate_midi_csv(&self, output_path: &str) -> std::io::Result<()> {
        let file = File::create(output_path)?;
        let mut writer = BufWriter::new(file);

        // MIDI header
        writeln!(writer, "0, 0, Header, 1, 3, 384")?;
        writeln!(writer, "1, 0, Start_track")?;
        writeln!(writer, "1, 0, Time_signature, 4, 2, 24, 8")?;
        writeln!(writer, "1, 0, Tempo, 500000")?;
        writeln!(writer, "1, {}, End_track", self.pointer_at as u32 * MULTI)?;

        // Main track
        writeln!(writer, "2, 0, Start_track")?;
        writeln!(writer, r#"2, 0, Text_t, "Decompressed MIDI""#)?;
        writeln!(writer, r#"2, 0, Title_t, "Main Track""#)?;

        // Note events
        for time in 0..self.pointer_at {
            for pitch in 0..87 {
                let current = self.notes[time][pitch];
                let previous = if time > 0 { self.notes[time-1][pitch] } else { false };

                if current && !previous {
                    writeln!(writer, "2, {}, Note_on_c, 1, {}, 127", 
                        time as u32 * MULTI, 
                        pitch + 21
                    )?;
                }
                if !current && previous {
                    writeln!(writer, "2, {}, Note_off_c, 1, {}, 0", 
                        time as u32 * MULTI, 
                        pitch + 21
                    )?;
                }
            }
        }

        // Footer
        writeln!(writer, "2, {}, End_track", self.pointer_at as u32 * MULTI)?;
        writeln!(writer, "0, 0, End_of_file")?;

        Ok(())
    }
}

fn main() -> std::io::Result<()> {
    let input_files = read_dir(INPUT_FOLDER)?;
    
    for entry in input_files {
        let entry = entry?;
        let input_path = entry.path();
        let filename = entry.file_name().into_string().unwrap();

        // Skip non-text files
        if !filename.ends_with(".cary") {
            continue;
        }

        let output_path = format!("{}/reconstructed_{}", OUTPUT_FOLDER, filename);

        println!("Processing: {}", filename);
        
        let mut decompressor = MidiDecompressor::new();
        match decompressor.load_compressed_file(input_path.to_str().unwrap()) {
            Ok(_) => {
                decompressor.generate_midi_csv(&output_path)?;
                println!("Successfully reconstructed: {}", filename);
            }
            Err(e) => eprintln!("Error processing {}: {}", filename, e),
        }
    }

    println!("Decompression complete!");
    Ok(())
}