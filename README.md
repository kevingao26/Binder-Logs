# Binder-Logs

## Part 1: EDA

Was done in the snap/common/jupyter folder.

Two main notebooks:
 - `Binder-Specs-EDA.ipynb` - Deals with `specs.sqlite` dataset, parses and builds a dependency table for network graph.
 - `Binder-Launches-EDA.ipynb` - Deals with `launches.db` dataset, for launch times / timestamp EDA.

Files needed:
 - `binder-launches.db`
 - `binder-specs.sqlite`
 
Generated files:
 - `dp_df.pkl` (specs, 8)
 - `edge_list_L1200_Dfull.pkl` (specs)
 - `cos_sim_dp_df.gexf` (specs, 16)

Dropped:
 - `Binder-EDA`: merging tables did not produce


-------------------------------------------------------------------------------------------------------------------------------------

### FACTORS beyond datasets:
 - [2, 3] Size of each package
 - [3] Package stats (stars, forks, etc.)
 - [2] Dependencies of package
 - [3] Version history
 - [2] Time to install package

### The DATA:
 - Individually collected data will be stored in the Part 2 and 3 folders that correspond to them, in a specified format (start and end index, set # for times). The combined data `combined_times.csv`, `combined_depsize.csv`, and `scraper_final.csv` will be stored in Part 4, where they are explored and cleaned in `pentagon_EDA.ipynb` to create the quintessential dataset, `[INSERT NAME]`

## Part 2: Package benchmarking
Workflow:
 - Started with `[psuedocode] environment_testing.ipynb`. This contains the command line arguments to build a virtual environment, install a library, and then clear the scene for repetition.
 - A shortcut, `[dropped] package_install_times--no-cache-dir.ipynb`, did not pan out.
 - `benchmarker.sh`: Conversion of `[psuedocode] environment_testing.ipynb` into a working bash script. Runs a benchmark on a single library and saves the time into a text file called `timetaken.txt` (overwritten each time).
 - `dependency_fetch.ipynb`: Fetches all the pip dependencies in a list (each unique element once), taken from `binder-clean.json`. In reality, this was run after `bench_loop.py`, but that is irrelevant. PRODUCED: `dep_list.txt`, a dependency list.
 - `bench_loop.py`: Loops through a specified number of libraries and calls the `benchmarker.sh` shell script on each one to get a time. Saves all the times into a dataframe, with the name specified by (start, end, set), where start and end are the starting and ending positions of the `dep_list.txt`, and the set is the i'th iteration or repetition of the library being benchmarked. 
 - `benchmarks_sorter.py`: Takes all benchmark csvs that are produced from `bench_loop.py`, and combine them (repeated rows are averaged). This produces a final csv called `combined_times.csv`, which should look like:
 
![image](https://user-images.githubusercontent.com/70555752/122136203-b1d62500-cddd-11eb-853a-e24b0946797d.png)

 - `depsize_sorter.py`: Takes all dependency/size csvs that are produced from `bench_loop.py`, and combine them. This produces a final csv called `combined_depsize.csv`.

Overnight run times:
![image](https://user-images.githubusercontent.com/70555752/123179577-6ea33400-d425-11eb-9ec5-7253b3510fcc.png)

EDIT: Many packages uninstall unsuccessfully, and this is indicated with an "ERROR" entry in the dependencies list from the `combined_depsize.csv` column. If "ERROR" appears, this package will be removed across all CSVs in part 4.


 
## Part 3: Scraping package sites (libraries.io, PYPI)
Because selenium's webdriver was used, chromedriver.exe is needed for the scraper. Thus, I wrote the scraper locally with `dep_list.txt` as the input list of packages to scrape, and scraped through both libraries.io and PYPI. A CSV is generated with [package stats, version history, and package size], name defined by scraper_(start, end), and these are combined later in the notebook to create a final CSV named `scraper_final.csv`.

Sidenote: at first I was going to just use PYPI, but I saw "repository size" on libraries.io so I scraped both. After cross checking with installed libs in the venv, it turned out that repository size was vastly different, but libraries.io is still pretty useful because often times, either libraries.io or PYPI has the github stats when the other doesn't.


## Part 4: Data Sorting I
Combines the data collection of parts 2 and 3 for an analysis notebook.

An overview of the input files:
 - Times -> `bench_loop.py` generates individual CSVs, and combined with `benchmarks_sorter.py` to create: `combined_times.csv`.
 - Dependencies/Sizes -> `bench_loop.py` generates individual CSVs, and combined with `depsize_sorter.py` to create: `combined_depsize.csv`.
 - Stats/Versions -> `pypi_scraper.ipynb` generates individual CSVs, and also combines them to create: `scraper_final.csv`.

 - `pentagon_conversion.ipynb` -> converts the inputs into a big combined dataset that has all columns, which is then chopped into a (final) dataset with columns that we want `q_df.txt`.

 - In `binder-clean-dependency-table.ipynb`, also takes the pip columns from `binder-clean.json`, inspired from `binder-specs.sqlite`, to create a dependency table but with cleaned up dependency names now as well as updating the dependency table to contain all of a dependency's dependencies (recursive). Saves this as `dependency_table.hdf5` to be loaded in Part 5.

The folder `raw_data` contains all the individually collected data before they were combined through a python script.

## Part 5: Data Sorting II & Model Initiation
 - `pentagon_EDA.ipynb` -> analysis & visualizations for the final dataset, `q_df.txt`. Converts `dependency_table.hdf5` into a final `dependency_table_final.txt` that factors in package dependencies, and converts `q_df.txt` to a final `pentagon_df.txt`. Also creates a simple LANDLORD model that compares size / time: MODEL_naive_similarity: Similar to the LANDLORD model. Uses a parameter α as a threshold between image similarities (calculated by the cosine similarity between pandas columns) and groups recipes/images into sets for some graphs and results.

## Part 6: The Models
 - `Model Workspace.ipynb` - sandbox for writing model scripts. Converts the dependency table into one with just 0's and 1's.
 - Inputs: `binder.sqlite`, `pentagon_df.txt`, `dependency_table_final.h5`. 

*Scratched : Dask will be used in place of pandas to reduce runtimes.

The Models:
 - Naive Model w/ no caching - ...
 - Least Recently Used (LRU) - 
        - lru_i: only identical images are shared
        - lru_c: images that are fully covered by a container are shared
 - Base LANDLORD - ...
 - LANDLORD++ - ...
