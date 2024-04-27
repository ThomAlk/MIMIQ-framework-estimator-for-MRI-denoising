# MIMIQ Framework protocol estimator for image denoising

This project applies various filters to images for image processing tasks.

## Project Structure

- `filt/`: Contains the main code for the project.
  - `filters.py`: Contains the implementation of various image filters.
  - `imageFiltering.py`: Contains the `ImageFiltering` class for loading images and applying filters.
  - `parameters.py`: Contains the `Parameters` class for managing filter parameters.
  - `utils.py`: Contains utility functions for the project.
- `img_result/`: Contains the output images after applying filters.
- `input/`: Contains the input images for the project.
- `logs/`: Contains log files for the project.
- `main.py`: The main entry point for the project.
- `psnrs.csv`: Contains the peak signal-to-noise ratio (PSNR) values for the images.
- `requirements.txt`: Contains the Python dependencies for the project.
- `saved_models/`: Contains saved models for the project.

## Installation

### Linux

1. Clone the repository: `git clone https://github.com/yourusername/yourrepository.git`
2. Navigate to the project directory: `cd yourrepository`
3. Create a Python virtual environment: `python3 -m venv venv`
4. Activate the virtual environment: `source venv/bin/activate`
5. Install the dependencies: `pip install -r requirements.txt`

### Windows

1. Clone the repository: `git clone https://github.com/yourusername/yourrepository.git`
2. Navigate to the project directory: `cd yourrepository`
3. Create a Python virtual environment: `py -m venv venv`
4. Activate the virtual environment: `.\venv\Scripts\activate`
5. Install the dependencies: `pip install -r requirements.txt`

## Usage

### Linux

1. Activate the virtual environment: `source venv/bin/activate`
2. Run the main script: `python3 main.py`

### Windows

1. Activate the virtual environment: `.\venv\Scripts\activate`
2. Run the main script: `py main.py`
