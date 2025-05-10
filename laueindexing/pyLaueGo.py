#!/usr/bin/env python3

import numpy as np
import os
import argparse
import yaml
import subprocess as sub
from datetime import datetime
import traceback
from xml.etree import ElementTree
from xml.dom import minidom
import fire
from typing import Dict, Any, Optional, Union

from laueindexing.lau_dataclasses.step import Step
from laueindexing.lau_dataclasses.indexing import Indexing
from laueindexing.lau_dataclasses.config import LaueConfig

class PyLaueGo:
    def __init__(self, config=None, comm=None, supress_errors=False):
        """
        Initialize PyLaueGo with optional config and MPI communicator.
        
        Args:
            config: Configuration object or dictionary with configuration parameters.
                   If None, configuration will be loaded from command line arguments.
            comm: MPI communicator object for distributed processing.
        """
        self.parser = argparse.ArgumentParser()
        self.comm = comm
        self.supress_errors = supress_errors
        
        # Initialize config as None
        self._config = None
        
        # Set config if provided
        if config is not None:
            if isinstance(config, dict):
                self._config = LaueConfig.from_dict(config)
            elif isinstance(config, LaueConfig):
                self._config = config
            else:
                self._config = LaueConfig.from_dict(config)
                
        # For backward compatibility
        self.config = config

    def run_on_process(self):
        self.run(0, 1)

    def run(self, rank, size):
        """
        Run the LaueGo indexing process.
        
        Args:
            rank: MPI rank of current process (0 for main process)
            size: Total number of MPI processes
        """
        # Ensure we have a config object
        if self._config is None:
            self._config = self.parseArgs(description='Runs Laue Indexing.')
            
        # Set up error logging
        now = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
        self.errorLog = f"{self._config.outputFolder}/error/error_{now}.log"
        
        try:
            if rank == 0:
                self.createOutputDirectories()
            xmlSteps = []
            processFiles = self.getFilesByRank(rank, size)
            for filename in processFiles:
                step = self.processFile(filename)
                if step:
                    xmlStep = step.getXMLElem()
                    xmlSteps.append(xmlStep)

            if self.comm is not None:
                self.comm.Barrier()
                if rank != 0:
                    self.comm.send(xmlSteps, dest=0)
                else:
                    #recieve all collected steps from manager node and combine
                    for recv_rank in range(1, size):
                        xmlSteps += self.comm.recv(source=recv_rank)
                    xmlOut = os.path.join(self._config.outputFolder, f'{self._config.filenamePrefix}indexed.xml')
                    self.writeXML(xmlSteps, xmlOut)
            
            else:
                # Single thread implementation
                self.createOutputDirectories()
                xmlSteps = []
                processFiles = self.getAllFiles()

                for filename in processFiles:
                    step = self.processFile(filename)
                    if step:
                        xmlStep = step.getXMLElem()
                        xmlSteps.append(xmlStep)
                xmlOut = os.path.join(self._config.outputFolder, f'{self._config.filenamePrefix}indexed.xml')
                self.writeXML(xmlSteps, xmlOut)
                
        except Exception as e:
            if self.comm is not None:
                self.comm.Abort(1)

            if not self.supress_errors:
                raise e

            with open(self.errorLog, 'a') as f:
                f.write(traceback.format_exc())

    def parseArgs(self, description) -> LaueConfig:
        """
        Parse command line arguments and create a configuration object.
        
        This method is optional if a config was provided during initialization.
        
        Args:
            description: Description string for the argument parser.
            
        Returns:
            LaueConfig object with parsed configuration
        """
        # Return existing config if it's already set
        if self._config is not None:
            return self._config
            
        # Parse from command line if no config was provided
        if self.config is None:
            self.parser.add_argument(f'--configFile', dest='configFile', required=True)
            self.parser.add_argument(f'--saveTxt', dest='saveTxt', action='store_true')

            args = self.parser.parse_known_args()[0]
            with open(args.configFile) as f:
                config_dict = yaml.safe_load(f)
        else:
            config_dict = self.config
            
        # Add arguments to parser for command-line overrides
        for arg in config_dict:
            self.parser.add_argument(f'--{arg}', dest=arg, type=str, default=config_dict.get(arg))
            
        # Parse arguments
        args = self.parser.parse_args()
        
        # Update config with command-line arguments
        for arg_name, arg_value in vars(args).items():
            if arg_value is not None:
                config_dict[arg_name] = arg_value
                
        # Create and store the config object
        self._config = LaueConfig.from_dict(config_dict)
        
        # For backward compatibility
        self.args = args
        
        return self._config

    def createOutputDirectories(self):
        ''' set up output directory structure '''
        outputDirectories = ['peaks', 'p2q', 'index', 'error']
        if not os.path.exists(self._config.outputFolder):
            os.mkdir(self._config.outputFolder)
        for dir in outputDirectories:
            fullPath = os.path.join(self._config.outputFolder, dir)
            if not os.path.exists(fullPath):
                os.mkdir(fullPath)

    def getFilesByRank(self, rank, size):
        ''' divide files to process into batches among nodes '''
        if rank == 0:
            scanPoint = None
            depthRange = None
            if self._config.scanPointStart is not None and self._config.scanPointEnd is not None:
                scanPoint = np.arange(int(self._config.scanPointStart), int(self._config.scanPointEnd))
            if self._config.depthRangeStart is not None and self._config.depthRangeEnd is not None:
                depthRange = np.arange(int(self._config.depthRangeStart), int(self._config.depthRangeEnd))
            filenames = self.getInputFileNamesList(depthRange, scanPoint)
        else:
            filenames = None
        if self.comm is not None:
            filenames = self.comm.bcast(filenames, root = 0)
        nFiles = int(np.ceil(len(filenames) / size))
        start = rank * nFiles
        end = min(start + nFiles, len(filenames))
        processFiles = filenames[start:end]
        return processFiles
    
    def getAllFiles(self):
        """Get all files for processing based on configuration."""
        scanPoint = None
        depthRange = None
        
        if self._config.scanPointStart is not None and self._config.scanPointEnd is not None:
            scanPoint = np.arange(int(self._config.scanPointStart), int(self._config.scanPointEnd))
        if self._config.depthRangeStart is not None and self._config.depthRangeEnd is not None:
            depthRange = np.arange(int(self._config.depthRangeStart), int(self._config.depthRangeEnd))
            
        return self.getInputFileNamesList(depthRange, scanPoint)



    def getInputFileNamesList(self, depthRange=None, scanPoint=None):
        ''' generate the list of files name for analysis '''
        fnames = []
        if depthRange is not None and scanPoint is not None:
            for ii in range(len(scanPoint)):
                for jj in range(len(depthRange)):
                    fname = f'{self._config.filenamePrefix}{scanPoint[ii]}_{depthRange[jj]}.h5'
                    if os.path.isfile(os.path.join(self._config.filefolder, fname)):
                        fnames.append(fname)
        elif scanPoint is not None:
            for ii in range(len(scanPoint)):
                # if the depthRange exist
                fname = f'{self._config.filenamePrefix}{scanPoint[ii]}.h5'
                if os.path.isfile(os.path.join(self._config.filefolder, fname)):
                    fnames.append(fname)
        else:
            #process them all
            for root, dirs, files in os.walk(self._config.filefolder):
                for name in files:
                    if name.endswith('h5'):
                        fnames.append(name)
        #fileCount = len(fnames)
        #estTime = fileCount * 3.5 / 100
        #print(f"Estimated time to completion: {estTime} minutes for {fileCount} files")
        return fnames

    def processFile(self, filename):
        '''
        Each processing step is run as a C program which
        outputs results to a file that becomes the input to
        the next processing step
        '''
        step = self.parseInputFile(filename)
        peakSearchOut = self.peakSearch(filename)
        self.parsePeaksFile(peakSearchOut, step)
        step.detector.set('cosmicFilter', self._config.cosmicFilter)
        step.detector.set('geoFile', self._config.geoFile)
        step.detector.peaksXY.set('peakProgram', os.path.basename(self._config.peaksearchPath))
        # p2q will fail if there are no peaks
        Npeaks = step.detector.peaksXY.Npeaks
        if Npeaks > 0:
            p2qOut = self.p2q(filename, peakSearchOut)
            self.parseP2QFile(p2qOut, step.detector.peaksXY)
        # must have at least 2 peaks to index
        if Npeaks > 1:
            indexOut = self.index(filename, p2qOut)
            step.indexing = self.parseIndexFile(indexOut, Npeaks)
        else:
            step.indexing = Indexing()
            step.indexing.set('Nindexed', 0)
        step.indexing.set('indexProgram', os.path.basename(self._config.indexingPath))

        return step

    def parseInputFile(self, filename):
        ''' parse input h5 file '''
        filename = os.path.join(self._config.filefolder, filename)
        step = Step()
        step.fromH5(filename)
        return step

    def peakSearch(self, filename):
        '''
        /data34/JZT/server336/bin/peaksearch
        USAGE:  peaksearch [-b boxsize -R maxRfactor -m min_size -M max_peaks -s minSeparation -t threshold -T thresholdRatio -p (L,G) -S -K maskFile -D distortionMap] InputImagefileName  OutputPeaksFileName
        switches are:
        	-b box size (half width)
        	-R maximum R factor
        	-m min size of peak (pixels)
        	-M max number of peaks to examine(default=50)
        	-s minimum separation between two peaks (default=2*boxsize)
        	-t user supplied threshold (optional, overrides -T)
        	-T threshold ratio, set threshold to (ratio*[std dev] + avg) (optional)
        	-p use -p L for Lorentzian (default), -p G for Gaussian
        	-S smooth the image
        	-K mask_file_name (use pixels with mask==0)
        	-D distortion map file name
        '''
        peakSearchOutDirectory = os.path.join(self._config.outputFolder, 'peaks')
        peakSearchOutFile = os.path.join(peakSearchOutDirectory, f'peaks_{filename[:-3]}.txt')
        fullPath = os.path.join(self._config.filefolder, filename)
        cmd = [self._config.peaksearchPath, '-b', str(self._config.boxsize), '-R', str(self._config.maxRfactor),
            '-m', str(self._config.min_size), '-s', str(self._config.min_separation), '-t', str(self._config.threshold),
            '-p', self._config.peakShape, '-M', str(self._config.max_peaks)]
        if self._config.maskFile:
            cmd += ['-K', self._config.maskFile]
        if self._config.thresholdRatio != -1:
            cmd += ['-T', self._config.thresholdRatio]
        if self._config.smooth:
            cmd += ['-S', '']
        cmd += [fullPath, peakSearchOutFile]
        self.runCmdAndCheckOutput(cmd)
        return peakSearchOutFile

    def parsePeaksFile(self, peaksFile, step):
        '''
        Peak search command outputs a txt file in the form
        $attr1 val1
        $attr2 val2
        ...
        followed by a matrix with the values listed here as peakListAttrsNames
        '''
        with open(peaksFile, encoding='windows-1252', errors='ignore') as f:
            lines = f.readlines()
            for line in lines:
                line = line.split('\t')
                vals = []
                for val in line:
                    if val and not val.startswith('//'):
                        vals.append(val.strip().replace('$', ''))
                if len(vals) == 2:
                    step.set(vals[0], vals[1])
                elif vals:
                    vals = vals[0].split()
                    step.detector.peaksXY.addPeak(*vals)

    def p2q(self, filename, peakSearchOut):
        '''
        converting the peaks in XY detector to the G unit vectors
        !/data34/JZT/server336/bin/pixels2qs
        usage below
        pixels2qs [-g geometryFile] -x crystalDescriptionFile input_peaks_file output_qs_file
        	input_peaks_file    result from peak search
        	output_qs_file      name for new output file that holds the result
        	switches are:
        		-g geometry file (defaults to geometry.txt)
        		-x crystal description file
        !/data34/JZT/server336/bin/pixels2qs -g './KaySong/geoN_2021-10-26_18-28-16.xml' -x './KaySong/Fe.xml' 'temp.txt' 'temp_Peaks2G.txt'
        '''
        p2qOutDirectory = os.path.join(self._config.outputFolder, 'p2q')
        p2qOutFile = os.path.join(p2qOutDirectory, f'p2q_{filename[:-3]}.txt')
        cmd = [self._config.p2qPath, '-g', self._config.geoFile, '-x', self._config.crystFile, peakSearchOut, p2qOutFile]
        self.runCmdAndCheckOutput(cmd)
        return p2qOutFile

    def parseP2QFile(self, p2qFile, peaksXY):
        '''
        P2Q outputs a txt file
        Here we only want to parse out the values qX qY qZ
        which are listed in a matrix part way through the file
        '''
        with open(p2qFile, encoding='windows-1252', errors='ignore') as f:
            lines = f.readlines()

        getLine = False
        for line in lines:
            if getLine:
                line = line.split(', ')[:3]
                peaksXY.addQVector(*line)
            if '$N_Ghat+Intens' in line:
                #The values we care about are listed after $N_Ghat+Intens
                getLine = True

    def index(self, filename, p2qOut):
        '''
        # indexing the patterns
        #!/data34/JZT/server336/bin/euler

        #switches are:
        #	-k keV calc max
        #	-t keV test max
        #	-f filename with input peak data
        #	-o filename for output results
        #	-h  h k l (preferred hkl, three agruments following -h)
        #	-q suppress most terminal output (-Q forces output)
        #	-c cone angle (deg)
        #	-a angle tolerance (deg)
        #	-n max num. of spots from data file to use
        #	-i tagged value file with inputs, available tags are:
        #		$EulerInputFile	always include this tag to identify the file type
        #		$keVmaxCalc	maximum energy to calculate (keV)  [-k]
        #		$keVmaxTest	maximum energy to test (keV)  [-t]
        #		$inFile		name of file with input peak positions  [-f]
        #		$outFile	output file name  [-o]
        #		$hkl		preferred hkl, 3 space separated numbers, hkl toward detector center,  [-h]
        #		$quiet		1 or 0  [-q or -Q]
        #		$cone		cone angle from hkl(deg)  [-c]
        #		$angleTolerance angular tolerance (deg)  [-a]
        #		$maxData	maximum number of peaks  [-n]
        #		$defaultFolder	default folder to prepend to file names
        #!/data34/JZT/server336/bin/euler -q -k 30.0 -t 35.0 -a 0.12 -h 0 0 1 -c 72.0 -f 'temp_Peaks2G.txt' -o 'temp_4Index.txt'
        '''
        indexOutDirectory = os.path.join(self._config.outputFolder, 'index')
        indexOutFile = os.path.join(indexOutDirectory, f'index_{filename[:-3]}.txt')
        cmd = [self._config.indexingPath, '-q', '-k', str(self._config.indexKeVmaxCalc), '-t', str(self._config.indexKeVmaxTest),
            '-a', str(self._config.indexAngleTolerance), '-c', str(self._config.indexCone), '-f', p2qOut, '-h',
            str(self._config.indexH), str(self._config.indexK), str(self._config.indexL), '-o', indexOutFile]
        if not self.runCmdAndCheckOutput(cmd):
            return indexOutFile

    def parseIndexFile(self, indexFile, Npeaks):
        '''
        Index command outputs a txt file in the form
        $attr1 val1
        $attr2 val2
        ...
        followed by a matrix with the values listed here as indexListAttrsNames
        '''
        indexing = Indexing()
        with open(indexFile, encoding='windows-1252', errors='ignore') as f:
            lines = f.readlines()
            for line in lines:
                line = line.split('\t')
                vals = []
                for val in line:
                    if val and not val.startswith('//'):
                        val = val.strip()
                        if val.startswith('$'):
                            val = val.replace('$', '')
                        vals.append(val)
                if len(vals) == 2:
                    indexing.set(vals[0], vals[1])
                elif vals and vals[0].startswith('['):
                    indexing.patterns[-1].hkl_s.fromString(vals[0])
        indexing.set('Npeaks', Npeaks)
        return indexing

    def runCmdAndCheckOutput(self, cmd):
        '''
        Handle errors for subprocesses
        Some errors from subprocesses should not be fatal
        e.g. peak search throws error if no peaks found
        Continue processing and output whichever information
        was found for that step
        '''
        try:
            output = sub.check_output(cmd, stderr=sub.STDOUT)
        except sub.CalledProcessError as e:
            with open(self.errorLog, 'a') as f:
                f.write(e.output.decode())
            return e.returncode

    def writeXML(self, xmlSteps, xmlOutFile):
        ''' Create the combined XML tree and write to file '''
        allSteps = ElementTree.Element('AllSteps')
        for step in xmlSteps:
            allSteps.append(step)
        roughString = ElementTree.tostring(allSteps, short_empty_elements=False)
        parsed = minidom.parseString(roughString)
        prettyXML = parsed.toprettyxml(indent='    ')
        with open(xmlOutFile, 'w') as f:
            f.write(prettyXML)

def run_mpi(config=None):
    """
    Run PyLaueGo with MPI support.
    
    Args:
        config: Configuration dictionary or path to config file
    """
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    start = datetime.now()
    
    # Convert config file path to config dict if needed
    if isinstance(config, str) and os.path.exists(config):
        with open(config, 'r') as f:
            config_dict = yaml.safe_load(f)
    else:
        config_dict = config
            
    pyLaueGo = PyLaueGo(config_dict, comm)
    pyLaueGo.run(rank, size)
    
    if rank == 0:
        print(f'runtime is {datetime.now() - start}')


if __name__ == '__main__':
    fire.Fire(run_mpi)
