use std::fs::{File, read_dir};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

// Constants
const INPUT_DIR: &str = "../data/output/cary/";
const OUTPUT_DIR: &str = "../data/output/midicsv/";
const TIME_QUANTUM: u32 = 40;  // Same as compressor's quantization
const PITCH_RANGE: usize = 87;  // 87 notes (MIDI 21-107)
const MAX_TIME_STEPS: usize = 20_000;

struct MidiDecompressor {
    note_matrix: Vec<[bool; PITCH_RANGE]>,  // Time × Pitch matrix
    current_time_step: usize,
}

impl MidiDecompressor {
    fn new() -> Self {
        MidiDecompressor {
            note_matrix: vec![[false; PITCH_RANGE]; MAX_TIME_STEPS],
            current_time_step: 0,
        }
    }

    fn load_compressed_file(&mut self, file_path: &Path) -> std::io::Result<()> {
        self.reset_state();
        
        let file = File::open(file_path)?;
        let reader = BufReader::new(file);

        for line in reader.lines() {
            let line = line?;
            self.process_compressed_line(&line);
        }

        // Finalize the last time step
        self.current_time_step += 1;
        
        Ok(())
    }

    fn process_compressed_line(&mut self, line: &str) {
        for c in line.chars() {
            match c {
                ' ' => self.current_time_step += 1,
                '\n' => (),  // Ignore newlines (handled by BufReader)
                _ => self.process_note_char(c),
            }
        }
    }

    fn process_note_char(&mut self, c: char) {
        let pitch = c as i32 - 32 - 1;  // Convert ASCII to pitch index
        if (0..PITCH_RANGE as i32).contains(&pitch) {
            self.note_matrix[self.current_time_step][pitch as usize] = true;
        }
    }

    fn generate_midi_csv(&self, output_path: &Path) -> std::io::Result<()> {
        let file = File::create(output_path)?;
        let mut writer = BufWriter::new(file);

        self.write_midi_header(&mut writer)?;
        self.write_note_events(&mut writer)?;
        self.write_midi_footer(&mut writer)?;

        Ok(())
    }

    fn write_midi_header(&self, writer: &mut BufWriter<File>) -> std::io::Result<()> {
        writeln!(writer, "0, 0, Header, 1, 3, 384")?;
        writeln!(writer, "1, 0, Start_track")?;
        writeln!(writer, "1, 0, Time_signature, 4, 2, 24, 8")?;
        writeln!(writer, "1, 0, Tempo, 500000")?;
        writeln!(writer, "1, {}, End_track", self.current_time_step as u32 * TIME_QUANTUM)?;
        writeln!(writer, "2, 0, Start_track")?;
        writeln!(writer, r#"2, 0, Text_t, "Decompressed MIDI""#)?;
        writeln!(writer, r#"2, 0, Title_t, "Main Track""#)?;
        
        Ok(())
    }

    fn write_note_events(&self, writer: &mut BufWriter<File>) -> std::io::Result<()> {
        for time in 0..self.current_time_step {
            for pitch in 0..PITCH_RANGE {
                let current_note = self.note_matrix[time][pitch];
                let previous_note = time > 0 && self.note_matrix[time-1][pitch];

                match (current_note, previous_note) {
                    (true, false) => self.write_note_on(writer, time, pitch)?,
                    (false, true) => self.write_note_off(writer, time, pitch)?,
                    _ => (),
                }
            }
        }
        Ok(())
    }

    fn write_note_on(&self, writer: &mut BufWriter<File>, time: usize, pitch: usize) -> std::io::Result<()> {
        writeln!(
            writer,
            "2, {}, Note_on_c, 1, {}, 127",
            time as u32 * TIME_QUANTUM,
            pitch + 21  // Convert to MIDI note number
        )
    }

    fn write_note_off(&self, writer: &mut BufWriter<File>, time: usize, pitch: usize) -> std::io::Result<()> {
        writeln!(
            writer,
            "2, {}, Note_off_c, 1, {}, 0",
            time as u32 * TIME_QUANTUM,
            pitch + 21  // Convert to MIDI note number
        )
    }

    fn write_midi_footer(&self, writer: &mut BufWriter<File>) -> std::io::Result<()> {
        writeln!(writer, "2, {}, End_track", self.current_time_step as u32 * TIME_QUANTUM)?;
        writeln!(writer, "3, 0, Start_track")?;
        writeln!(writer, "3, 0, Title_t, \"MIDI\"")?;
        writeln!(writer, "3, 1536, End_track")?;
        writeln!(writer, "0, 0, End_of_file")?;
        Ok(())
    }

    fn reset_state(&mut self) {
        self.note_matrix.fill([false; PITCH_RANGE]);
        self.current_time_step = 0;
    }
}

fn main() -> std::io::Result<()> {
    let input_dir = read_dir(INPUT_DIR)?;
    
    for entry in input_dir {
        let entry = entry?;
        let input_path = entry.path();
        
        // Skip non-cary files
        if !input_path.extension().map_or(false, |ext| ext == "cary") {
            continue;
        }

        let filename = input_path.file_name().unwrap().to_string_lossy();
        let output_path = Path::new(OUTPUT_DIR).join(format!("reconstructed_{}.csv", filename));

        println!("Processing: {}", filename);
        
        let mut decompressor = MidiDecompressor::new();
        match decompressor.load_compressed_file(&input_path) {
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