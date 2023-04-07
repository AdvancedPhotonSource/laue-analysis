# laue-indexing

Programs to index laue outputs at beamline 34IDE.

Status: Ported underlying programs and control software to Polaris software stack.

## To Run

### Set Up Environment

**Processing environment should include:**
h5py
numpy
yaml (pyyaml)
mpi4py
module load gsl

### Compile C Programs

If running on a new machine, compile the 3 C sub programs in the euler, peaksearch, and pixels2qs directories. There is a makefile included for each.

### Modify Config File

An example config file is included in `config/defaults.yml`
Config values will be different based on experiment so **these should be reviewed** before processing each data set.

To process a partial data set, include values scanPointStart, scanPointEnd, depthRangeStart, depthRangeEnd in the config file
If these values are not included, all files in the input directory will be processed.

Set the value for **outputFolder** as the full path where the processed output files will be written.
Files will be output to the outputFolder in the structure:

```
input_dir_name.xml - this is the final processed result
error_current_timestamp.log - any errors logged here
index/index_out_0_0.txt - results of the index subprocess for each point
index/index_out_0_1.txt
...
p2q/p2q_out_0_0.txt - results of the p2q subprocess for each point
p2q/p2q_out_0_1.txt
...
peaks/peaks_out_0_0.txt - results of the peak search subprocess for each point
peaks/peaks_out_0_1.txt
```

Set the value for **filefolder** to the full path of the directory containing the input h5 files.

Set the value for **pathbins** to the top level directory containing the euler, peaksearch, and pixels2qs programs.
Unless you've moved them, it will be the current directory of this readme.

**Upload the geometry file** for this experiment and set the value for **geoFile** to the full path to the geometry file.

**Upload the crystal structure file** for this experiment and set the value for **crystFile** to the full path to the crystal structure file.

### Run the Program

Values from the config file can also be modified at runtime from the command line

e.g. `--outputFolder /eagle/APSDataAnalysis/hparraga/output`

Path to the config file must be included.

```
mpirun -np 32 pyLaueGo.py --configFile {path to config file}
```

For a small number of files, lower the number of mpi processes with the `-np` flag

For more information on what arguments are available, there is also a `--help` flag