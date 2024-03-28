# Project Structure
The project is structured as follows:

- main_run_all.py: The main script that orchestrates the data loading, processing, and visualization.
- algo_functions.py: Contains algorithmic functions for data analysis.
- loading.py: Handles the loading of data from the data folder.
- visuals.py: Generates plots and saves them in the plots folder.
- data/: A directory containing all the data files necessary to run the project.
- plots/: The directory where generated plots are saved. This folder houses the latest versions of each plot.
- run_results.txt: A text file containing hand-written copies of the print outputs from the runs.
- populations.txt: A text file with hand-written documentation of populations as analyzed in the project.


# Getting Started
To get this project running on your local machine, follow these steps:

## Prerequisites
Ensure you have Python installed on your machine. This project was developed with Python 3.9.13. You may need to install additional packages, which are listed in the requirements.txt file.

## Setup
Clone this repository to your local machine.
Install the required packages using pip:
### Copy code
pip install -r requirements.txt


# Running the Project
Execute the main script from the terminal or command prompt:

### Copy code
python main_run_all.py
This will kick off the analysis process, utilizing data from the data directory and generating plots that will be saved in the plots directory.