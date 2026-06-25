import io
import zipfile
from tqdm import tqdm
from datasets import load_dataset

def getTinyStoriesDataset():
  # 1. Load the TinyStories dataset from Hugging Face
    print("Downloading dataset...")
    dataset = load_dataset("roneneldan/TinyStories")

    # 2. Open a text file in write mode with UTF-8 encoding
    output_file = "tinystories.txt"
    print(f"Writing stories to {output_file}...")

    with open(output_file, "w", encoding="utf-8") as f:
        # 3. Iterate through the 'train' split of the dataset
        for item in dataset["train"]:
            story = item["text"].strip()
            # Write the story followed by two newlines to separate them
            f.write(story + "\n\n")

    print("Finished successfully!")


def getPartTinyStoriesDataset(
    input_file_path: str, 
    output_file_path: str, 
    line_limit: int=10_1000):
    """
    Gets part of the "TinyStories" dataset
    To use it invoke the following:
       getPartTinyStoriesDataset("tinystories.txt", "tinystories-10000.txt")    
    param: input_file_path: Inout file part
    param: output_file_path: Output file part
    param: line_limit: Line limit
    """
  
    # Open both files: one for reading, one for writing
    with open(input_file_path, "r", encoding="utf-8") as infile, \
        open(output_file_path, "w", encoding="utf-8") as outfile:
        
        for i, line in enumerate(infile):
            if i >= line_limit:
                break
            
            # Write the line directly to the new file
            if len(line.strip()) > 0:
              current_line = f"{line.strip()}<|endoftext|>"
              outfile.write(current_line)

    print(f"Successfully saved the first {line_limit} lines to {output_file_path}")
    
def save_zipped_lines(
    zip_path: str,
    output_txt_path: str,
    max_lines: int,
    chunk_size: int = 1024 * 1024,
   ):
    """Reads a specified number of lines from a zipped file in chunks

    and saves them directly into a new text file.
    """
    line_count = 0
    leftover = ""

    # Open the destination file for writing
    with open(output_txt_path, "w", encoding="utf-8") as out_file:
        # Create a progress bar capped at the maximum requested lines
        with tqdm(
            total=max_lines,
            desc="Writing lines",
            unit=" lines",
            dynamic_ncols=True,
        ) as pbar:
            with zipfile.ZipFile(zip_path, "r") as archive:
                txt_filename = archive.namelist()[0]

                with archive.open(txt_filename, "r") as binary_file:
                    with io.TextIOWrapper(
                        binary_file, encoding="utf-8", errors="ignore"
                    ) as text_file:

                        while line_count < max_lines:
                            # 1. Read a block of text into memory
                            chunk = text_file.read(chunk_size)
                            if not chunk:
                                # End of source file reached early
                                if leftover:
                                    out_file.write(leftover + "\n")
                                    pbar.update(1)
                                break

                            # 2. Reconstruct lines using leftover from previous chunk
                            combined = leftover + chunk
                            lines = combined.split("\n")

                            # 3. Cache the final incomplete line split by chunk boundary
                            leftover = lines.pop()

                            # 4. Stream completed lines directly to the output file
                            for line in lines:
                                if line_count >= max_lines:
                                    break
                                
                                if len(line.strip()) > 0:
                                  current_line = f"{line.strip()}<|endoftext|>"
                                  # outfile.write(current_line)
                                  # current_line = f"{line.strip()}<|endoftext|>"
                                  # out_file.write(line + "\n")
                                  out_file.write(current_line)
                                  line_count += 1
                                pbar.update(1)
    print(f"\n\nSuccefully created {output_txt_path} file with {max_lines} lines")


