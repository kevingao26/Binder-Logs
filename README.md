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


## Part 2: Package benchmarking
Workflow:
 - Started with `[psuedocode] environment_testing.ipynb`. This contains the command line arguments to build a virtual environment, install a library, and then clear the scene for repetition.
 - A shortcut, `[dropped] package_install_times--no-cache-dir.ipynb`, did not pan out.
 - `benchmarker.sh`: Conversion of `[psuedocode] environment_testing.ipynb` into a working bash script. Runs a benchmark on a single library and saves the time into a text file called `timetaken.txt` (overwritten each time).
 - `dependency_fetch.ipynb`: Fetches all the pip dependencies in a list (each unique element once), taken from `binder-clean.json`. In reality, this was run after `bench_loop.py`, but that is irrelevant. PRODUCED: `dep_list.txt`, a dependency list.
 - `bench_loop.py`: Loops through a specified number of libraries and calls the `benchmarker.sh` shell script on each one to get a time. Saves all the times into a dataframe, with the name specified by (start, end, set), where start and end are the starting and ending positions of the `dep_list.txt`, and the set is the i'th iteration or repetition of the library being benchmarked. 
 - `benchmarks_sorter.py`: Takes all csvs that are produced from `bench_loop.py`, and combine them (repeated rows are averaged). This produces a final csv called `combined_times.csv`, which should look like:
 
![image](https://user-images.githubusercontent.com/70555752/122136203-b1d62500-cddd-11eb-853a-e24b0946797d.png)

Overnight run times:
![image](https://user-images.githubusercontent.com/70555752/122448998-14940180-cf41-11eb-88b2-efa5d4bc895f.png)


 
## Part 3: Scraping package sites (libraries.io, PYPI)
Because selenium's webdriver was used, chromedriver.exe is needed for the scraper. Thus, I wrote the scraper locally with `dep_list.txt` as the input list of packages to scrape, and scraped through both libraries.io and PYPI. A CSV is generated with [package stats, version history, and package size], name defined by scraper_(start, end), and these are combined later in the notebook to create a final CSV named `scraper_final.csv`.
