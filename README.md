# Flight Data Analysis Module (Advanced Programming for Data Science)
## Module Overview

This module was developed as part of a group project in the 'Advanced Programming for Data Science' course at NOVA School of Business and Economics. The module provides a comprehensive solution for analyzing and visualizing flight data. It includes the Flights class, designed for various tasks like plotting airports in countries, creating flight distance histograms, and visualizing flight paths. The class manages data download, preprocessing, and visualization with libraries like Pandas, Matplotlib, and Geopandas. It also incorporates OpenAI’s language models, which support the analysis with detailed descriptions and specifications.

Flights automates flight data retrieval, processes the data for enhanced utility (such as calculating distances and merging datasets), and provides methods for detailed visual analyses. These capabilities enable in-depth study of global flight operations, including distance analysis, and identifying popular airplane models.

## Requirements

Before you begin, ensure you have the following prerequisites installed on your system:
- Python 3
- conda or Miniconda

These are essential for creating and managing the project environment using the provided `environment.yaml` file.

## How to Run the Project

1. Clone the Repository
   ```sh
    git clone https://github.com/malte-beep/Flight-Data-Analysis.git
    cd Flight-Data-Analysis
    ```

3. Create and activate the Conda environment
  
  Use the provided `environment.yaml` file to create an environment with all necessary dependencies:
   ```sh
    $ conda env create -f environment.yaml
   ```
  Wait for Conda to set up the environment. Once it's done, you can activate it. With the environment active, all project dependencies are available for use.

3. Launch the `Showcase_Notebook.ipynb` notebook to view a brief showcase of the main functionalities of the module.

## Remarks

Setting the PEP 8 compliance threshold of pylint to 8 is a practical compromise that balances code quality with developer efficiency. It upholds Python's style guidelines while offering flexibility for cases where perfect adherence isn't feasible or required. This approach helps keep the code readable and maintainable without turning linting into a development bottleneck. It fosters a culture of quality while addressing the practical challenges of software development.

### Authors

- Antoine Thomas
- Lennie König
- Malte Haupt
- Noah Schültke
