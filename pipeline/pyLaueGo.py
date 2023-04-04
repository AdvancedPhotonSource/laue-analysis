#!/usr/bin/env python3

import h5py
import numpy as np
import os
import glob
import argparse
import yaml
import subprocess as sub
from datetime import datetime
from mpi4py import MPI

from xmlWriter import XMLWriter


class PyLaueGo:

    DEFAULT_ARGS_FILE = 'defaults.yml'

    def __init__(self):
        self.parser = argparse.ArgumentParser()
        with open(self.DEFAULT_ARGS_FILE) as f:
            defaults = yaml.safe_load(f)
        for arg in defaults:
            self.parser.add_argument(f'--{arg}', dest=arg, type=str, default=defaults.get(arg))

    def run(self, rank, size):
        #global args shared for all samples e.g. title
        globalArgs = self.parseArgs(description='Runs Laue Indexing.') #TODO better description
        try:
            scanPoint = np.arange(int(globalArgs.scanPointStart), int(globalArgs.scanPointEnd))
            depthRange = np.arange(int(globalArgs.depthRangeStart), int(globalArgs.depthRangeEnd))
            xmlWriter = XMLWriter()
            xmlSteps = []

            if rank == 0:
                filenames = self.getInputFileNamesList(depthRange, scanPoint, globalArgs)
            else:
                filenames = None
            filenames = comm.bcast(filenames, root = 0)
            nFiles = int(np.ceil(len(filenames) / size))
            start = rank * nFiles
            end = min(start + nFiles, len(filenames))
            processFiles = filenames[start:end]

            for filename in processFiles:
                sampleArgs = self.parseInputFile(filename, globalArgs)
                peakSearchOut = self.peakSearch(filename, sampleArgs)
                sampleArgs = self.parsePeaksFile(peakSearchOut, sampleArgs)
                p2qOut = self.p2q(filename, peakSearchOut, sampleArgs)
                sampleArgs = self.parseP2QFile(p2qOut, sampleArgs)
                indexOut = self.index(filename, p2qOut, sampleArgs)
                sampleArgs = self.parseIndexFile(indexOut, sampleArgs)
                xmlSteps.append(xmlWriter.getStepElement(sampleArgs))

            comm.Barrier()
            if rank != 0:
                comm.send(xmlSteps, dest=0)
            else:
                for recv_rank in range(1, size):
                    xmlSteps += comm.recv(source=recv_rank)
                xmlWriter.write(xmlSteps, globalArgs.xmlOutFile)
        except Exception as e:
            with open(globalArgs.errorLog, 'w') as f:
                f.write(e)
            comm.Abort(1)

    def parseArgs(self, description):
        ''' get user inputs and if user didn't input a value, get default value '''
        with open(self.DEFAULT_ARGS_FILE) as f:
            defaults = yaml.safe_load(f)
        self.parser.description = description
        args = self.parser.parse_args()

        for defaultArg in defaults:
            if not args.__dict__.get(defaultArg):
                setattr(args, defaultArg, defaults.get(defaultArg))
        return args

    def getInputFileNamesList(self, depthRange, scanPoint, args):
        ''' generate the list of files name for analysis '''
        allFiles = []
        for root, dirs, files in os.walk(path):
            for name in files:
                if name.endswith('h5'):
                    allFiles.append(name)
                    print(name)
        exit()
        fnames = []
        if depthRange and scanPoint:
            for ii in range(len(scanPoint)):
                # if the depthRange exist
                fname = f'{args.filenamePrefix}{scanPoint[ii]}'
                if os.path.isfile(f"{args.filefolder}{fname}.h5"):
                    fnames.append(fname)
        elif scanPoint:
            for ii in range(len(scanPoint)):
                for jj in range (len(depthRange)):
                    fname = f'{args.filenamePrefix}{scanPoint[ii]}_{depthRange[jj]}'
                    if os.path.isfile(f"{args.filefolder}{fname}.h5"):
                        fnames.append(fname)
        else:
            #process them all
            fnames = allFiles
        return fnames

    def clearSaveFolder(self, savefolder):
        ''' just for testing clean the folder before running the next cell,
            as it will not show up what is wrong '''
        files = glob.glob(savefolder + '*')
        for f in files:
            os.remove(f)

    def parseInputFile(self, filename, args):
        filename = args.filefolder + filename + '.h5'
        attrsNameMap = {
            'title': 'entry1/title',
            'sampleName': 'entry1/sample/name',
            'CCDshutter': 'entry1/microDiffraction/CCDshutter',
            'detectorID': 'entry1/detector/ID',
            'Nx': 'entry1/detector/Nx',
            'Ny': 'entry1/detector/Ny'
        }
        with h5py.File(filename, 'r') as f:
            for attr in attrsNameMap:
                a = f[attrsNameMap.get(attr)][0]
                if type(a) == np.bytes_:
                    a = a.decode('UTF-8')
                setattr(args, attr, a)
        setattr(args, 'date', os.path.getctime(filename))
        return args

    def peakSearch(self, filename, args):
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
        peakSearchPath = args.pathbins + 'peaksearch'
        peakSearchOut = f'{args.saveFolder}peaks_{filename}.txt'
        fullPath = args.filefolder + filename + '.h5'
        cmd = [peakSearchPath, '-b', str(args.boxsize), '-R', str(args.maxRfactor),
            '-m', str(args.min_size), '-s', str(args.min_separation), '-t', str(args.threshold),
            '-p', args.peakShape, '-M', str(args.max_peaks)]
        if args.maskFile:
            cmd += ['-K', args.maskFile]
        if args.smooth:
            cmd += ['-S', '']
        cmd += [fullPath, peakSearchOut]
        sub.run(cmd)

        return peakSearchOut

    def parsePeaksFile(self, peaksFile, args):
        ''' read the file made by peak search command '''
        peakListAttrsNames = ['fitX', 'fitY', 'intens', 'integral', 'hwhmX', 'hwhmY', 'tilt', 'chisq']
        peakListAttrs = [[] for _ in peakListAttrsNames]

        with open(peaksFile, encoding='windows-1252', errors='ignore') as f:
            lines = f.readlines()
            for line in lines:
                line = line.split('\t')
                vals = []
                for val in line:
                    if val and not val.startswith('//'):
                        vals.append(val.strip().replace('$', ''))
                if len(vals) == 2:
                    setattr(args, vals[0], vals[1])
                elif vals:
                    vals = vals[0].split()
                    for i in range(len(vals)):
                        peakListAttrs[i].append(vals[i])

        for i in range(len(peakListAttrs)):
            setattr(args, peakListAttrsNames[i], ' '.join(peakListAttrs[i]))
        return args

    def p2q(self, filename, peakSearchOut, args):
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
        p2qPath =  args.pathbins + 'pixels2qs'
        p2qOut = f'{args.saveFolder}p2q_{filename}.txt'
        cmd = [p2qPath, '-g', args.geoFile, '-x', args.crystFile, peakSearchOut, p2qOut]
        sub.run(cmd)
        return p2qOut

    def parseP2QFile(self, p2qFile, args):
        # extract the peaks from the xml file
        p2qListAttrsNames = ['qX', 'qY', 'qZ']
        p2qListAttrs = [[] for _ in p2qListAttrsNames]

        with open(p2qFile, encoding='windows-1252', errors='ignore') as f:
            lines = f.readlines()

        getLine = False
        for line in lines:
            if getLine:
                line = line.split()
                for i in range(len(p2qListAttrs)):
                    p2qListAttrs[i].append(line[i])

            if '$N_Ghat+Intens' in line:
                #The values we care about are listed after $N_Ghat+Intens
                getLine = True

        for i in range(len(p2qListAttrs)):
            setattr(args, p2qListAttrsNames[i], ' '.join(p2qListAttrs[i]))
        return args

    def index(self, filename, p2qOut, args):
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
        indexPath = args.pathbins + 'euler'
        indexOut = f'{args.saveFolder}index_{filename}.txt'
        cmd = [indexPath, '-q', '-k', str(args.indexKeVmaxCalc), '-t', str(args.indexKeVmaxTest),
            '-a', str(args.indexAngleTolerance), '-c', str(args.indexCone), '-f', p2qOut, '-h',
            str(args.indexH), str(args.indexK), str(args.indexL), '-o', indexOut]
        sub.run(cmd)

        return indexOut

    def parseIndexFile(self, indexFile, args):
        ''' parse the index file which has some vals as $abc 123 and
            some as a table with extra punctuation to remove '''
        indexListAttrsNames = ['h', 'k', 'l', 'PkIndex']
        indexListAttrs = [[] for _ in indexListAttrsNames]

        with open(indexFile, encoding='windows-1252', errors='ignore') as f:
            lines = f.readlines()
            for line in lines:
                line = line.split('\t')
                vals = []
                for val in line:
                    if val and not val.startswith('//'):
                        val = val.strip()
                        if val.startswith('$'):
                            val = val.replace('$', '').replace('0', '')
                        vals.append(val)
                if len(vals) == 2:
                    setattr(args, vals[0], vals[1])
                elif vals and vals[0].startswith('['):
                    vals = vals[0]
                    for c in '[]()':
                        vals = vals.replace(c, '')
                    vals = vals.split()
                    #$array0 10  14  G^ h k l intens E(keV) err(deg) PkIndex
                    for i in range(len(indexListAttrs) - 1):
                        indexListAttrs[i].append(vals[i + 4])
                    indexListAttrs[-1].append(vals[-1])

        for i in range(len(indexListAttrs)):
            setattr(args, indexListAttrsNames[i], ' '.join(indexListAttrs[i]))
        setattr(args, 'xtlFile', args.xtalFileName)
        args.latticeParameters = args.latticeParameters.replace('}', '').replace('{', '').strip()
        args = self.getRecipLatticeStar(args)
        args = self.getAtom(args)
        return args

    def getRecipLatticeStar(self, args):
        rl = args.recip_lattice
        rl = rl.replace('{', '').replace('}', ' ') #leave a space
        rl = rl.split()
        for i in range(len(rl)):
            rl[i] = rl[i].replace(',', ' ')
        setattr(args, 'astar', rl[0])
        setattr(args, 'bstar', rl[1])
        setattr(args, 'cstar', rl[2])
        return args

    def getAtom(self, args):
        #from AtomDesctiption1='{Ni001  0 0 0 1} to
        #<atom Ni=\'30\' label=\'Ni001\' n=\'1\' symbol=\'Ni\'>0 0 0</atom>\n
        setattr(args, 'Ni', args.NiData)
        atom = args.AtomDesctiption1.replace('}', '').replace('{', '').split()
        setattr(args, 'n', atom[-1])
        setattr(args, 'label', atom[0])
        setattr(args, 'atom', ' '.join(atom[1:3]))
        setattr(args, 'symbol', args.label[:2])
        return args

if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    start = datetime.now()
    pyLaueGo = PyLaueGo()
    pyLaueGo.run(rank, size)
    if rank == 0:
        print(f'runtime is {datetime.now() - start}')
