# Binder-Logs

## Part 1: EDA

Was done in the snap/common/jupyter folder.

Two main notebooks:
 - `Binder-Specs-EDA.ipynb` - Deals with specs.sqlite dataset, parses and builds a dependency table for network graph.
 - Binder-Launches-EDA.ipynb - Deals with launches.db dataset, for launch times / timestamp EDA.

Files needed:
 - binder-launches.db
 - binder-specs.sqlite
 
Generated files:
 - dp_df.pkl (specs, 8)
 - edge_list_L1200_Dfull.pkl (specs)
 - cos_sim_dp_df.gexf (specs, 16)

Dropped:
 - Binder-EDA: merging tables did not produce

## Part 2: Package benchmarking
Workflow:
 - Started with `[psuedocode] environment_testing.ipynb`.
 - 
