# Organism

organism trains a transformer model that predicts organism names.

## Model

The model is adapted from Andrej Karpathy's model in
[Let's build GPT: from scratch, in code, spelled out](https://youtu.be/kCc8FmEb1nY?si=IetCFKNF8Zs4u8dc).

## Training data

`data/organism_names.txt` was extracted from the [Darwin Core database](https://dwc.tdwg.org/). The names were taken
from the `scientificName` column of the `NameUsage.tsv` table. The file was then truncated to be less than 100 MB. Also,
names containing non-ASCII characters were excluded.
