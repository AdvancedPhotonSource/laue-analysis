import h5py
import numpy as np
import scipy.io as sio
import csv
import subprocess as sub
import re


def fn_Jonsh5structure():
    data2D = np.zeros((2048,2048))
    sampleX = 0.0
    sampleY = 0.0;
    sampleZ = 0.0;
    dicth5 = {'Facility/facility_beamline': "34ID-E", 
          'Facility/facility_float'   : 3.1415,
          'Facility/facility_int'     : 12,
          'Facility/facility_name'    : "APS",
          'Facility/facility_sector'  : "Xor/UNI",
          'Facility/facility_station' : "E",
          'entry1/data/data'          : data2D,
          'entry1/detector/ID'        : "PE1621 723-3335",
          'entry1/detector/Nx'        : 2048,
          'entry1/detector/Ny'        : 2048,
          'entry1/detector/binx'      : 1,
          'entry1/detector/biny'      : 1,
          'entry1/detector/detID'     : 751,
          'entry1/detector/endx'      : 2047,
          'entry1/detector/endy'      : 2047,
          'entry1/detector/exposure'  : 0.25,
          'entry1/detector/gain'      : 2.0,
          'entry1/detector/make'      : "Perkin Elmer",
          'entry1/detector/model'     : "XRD0820",
          'entry1/detector/startx'    : 0,
          'entry1/detector/starty'    : 0,
          'entry1/microDiffraction/BeamBad'           : 0,
          'entry1/microDiffraction/CCDshutter'        : 1,
          'entry1/microDiffraction/HutchTemperature'  : 23.183,
          'entry1/microDiffraction/LightOn'           : 0,
          'entry1/microDiffraction/MonoMode'          : "white slitted",
          'entry1/microDiffraction/Wslit_height'      : 0.08,
          'entry1/microDiffraction/Wslit_width'       : 0.06,
          'entry1/microDiffraction/filter/description': 1,
          'entry1/microDiffraction/source/L32_height' : 5,
          'entry1/microDiffraction/source/L32_width'  : 0.098,
          'entry1/microDiffraction/source/current'    : 102.2,
          'entry1/microDiffraction/source/distance'   : 65,
          'entry1/microDiffraction/source/gap'        : 18.47199,
          'entry1/microDiffraction/source/name'       : "Undulator_#34s_3.3cm",
          'entry1/microDiffraction/source/probe'      : "x-ray",
          'entry1/microDiffraction/source/taper'      : 4.99,
          'entry1/microDiffraction/source/top_up'     : 1,
          'entry1/microDiffraction/source/type'       : "Synchrotron X-ray Source",
          'entry1/monitor/I0'                         : 70871.0,
          'entry1/monitor/I0_calc'                    : 70871.0,
          'entry1/monitor/I_final'                    : 1155.0,
          'entry1/monitor/I_final_calc'               : 1155.0,
          'entry1/monitor/I_start'                    : 244915.0,
          'entry1/monitor/I_start_calc'               : 244915.0,
          'entry1/monitor/ScalerClockFreq'            : 10000000,
          'entry1/monitor/ScalerClock_calc'           : 10000000,
          'entry1/monitor/ScalerCountTime'            : 1.0,
          'entry1/monitor/mode'                       : "timer",
          'entry1/sample/distance'                    : 0.0,
          'entry1/sample/incident_energy'             : 20.0,
          'entry1/sample/name'                        : "",
          'entry1/sample/sampleX'                     : sampleX,
          'entry1/sample/sampleY'                     : sampleY,
          'entry1/sample/sampleZ'                     : sampleZ,
          'entry1/scanNum'                            : 272544,
          'entry1/title'                              : "",
          'entry1/user/name'                          : "staff",
      }
    subs = list(dicth5.keys())
    val = list(dicth5.values())
    return subs, val, dicth5


def fn_genericJonsH5(filename, filenamesave, x1,x2,y1,y2, depth, data_2D):

    

    [subs, val, dicth5] = fn_Jonsh5structure()
    # to save temp.h5 for searching peaks
    f2 = h5py.File(filename, 'r')
    sampleX = f2['entry1/sample/sampleX'][0];
    sampleY = f2['entry1/sample/sampleY'][0];
    sampleZ = f2['entry1/sample/sampleZ'][0];
    for tt in range(len(subs)):
        dicth5[subs[tt]] = f2[subs[tt]]
    dicth5['entry1/data/data'] = data_2D
    dicth5['entry1/sample/sampleX'] = np.float64(sampleX)
    dicth5['entry1/sample/sampleY'] = np.float64(sampleY)
    dicth5['entry1/sample/sampleZ'] = np.float64(sampleZ)
    dicth5['entry1/detector/Nx'] = np.float64(np.shape(dicth5['entry1/data/data'])[0])
    dicth5['entry1/detector/Ny'] = np.float64(np.shape(dicth5['entry1/data/data'])[1])
    dicth5['entry1/detector/startx'] = np.float64(x1);
    dicth5['entry1/detector/starty'] = np.float64(y1);
    dicth5['entry1/detector/endx'] = np.float64(x2);
    dicth5['entry1/detector/endy'] = np.float64(y2);

    subs = list(dicth5.keys())
    val = list(dicth5.values())
    with h5py.File(filenamesave, 'w') as f:
        f.attrs.create('file_name', filenamesave)
        f.attrs.create('file_time', '2021-11-13')
        for kk in range(len(subs)):
            f.create_dataset(subs[kk], data=val[kk])
        f.create_dataset('entry1/timingClock', data = np.float64(1))
        f.create_dataset('entry1/depth', data = np.float64(depth))
        f.create_dataset('entry1/timingSignal', data = np.float64(1))
        f['entry1/sample/incident_energy'].attrs.create('units', 'keV')
        f['entry1/depth'].attrs.create('units', 'micron')
        f['entry1/sample/distance'].attrs.create('units', 'mm')
        f['entry1/sample/sampleX'].attrs.create('units', 'micron')
        f['entry1/sample/sampleY'].attrs.create('units', 'micron')
        f['entry1/sample/sampleZ'].attrs.create('units', 'micron')

    f2.close()
    #print(filenamesave + ' saved')
    
def fn_genericJonsH5_from2Ddata(filenamesave, x1,x2,y1,y2, depth, data_2D):

    
    Nx_key = 1
    [subs, val, dicth5] = fn_Jonsh5structure()
    # to save temp.h5 for searching peaks
    dicth5['entry1/data/data'] = data_2D
    dicth5['entry1/detector/startx'] = np.float64(x1);
    dicth5['entry1/detector/starty'] = np.float64(y1);
    dicth5['entry1/detector/endx'] = np.float64(x2);
    dicth5['entry1/detector/endy'] = np.float64(y2);
    
    try:
        dicth5['entry1/detector/Nx'] = np.float64(np.shape(dicth5['entry1/data/data'])[0])
        dicth5['entry1/detector/Ny'] = np.float64(np.shape(dicth5['entry1/data/data'])[1])
    except:
        Nx_key = 0;

    subs = list(dicth5.keys())
    val = list(dicth5.values())
    with h5py.File(filenamesave, 'w') as f:
        f.attrs.create('file_name', filenamesave)
        f.attrs.create('file_time', '2021-11-13')
        for kk in range(len(subs)):
            f.create_dataset(subs[kk], data=val[kk])
        f.create_dataset('entry1/timingClock', data = np.float64(1))
        f.create_dataset('entry1/depth', data = np.float64(depth))
        f.create_dataset('entry1/timingSignal', data = np.float64(1))
        f['entry1/sample/incident_energy'].attrs.create('units', 'keV')
        f['entry1/depth'].attrs.create('units', 'micron')
        f['entry1/sample/distance'].attrs.create('units', 'mm')
        f['entry1/sample/sampleX'].attrs.create('units', 'micron')
        f['entry1/sample/sampleY'].attrs.create('units', 'micron')
        f['entry1/sample/sampleZ'].attrs.create('units', 'micron')
        if Nx_key == 0:
            f.create_dataset('entry1/detector/Nx', data = np.float64(np.shape(dicth5['entry1/data/data'])[1]))
            f.create_dataset('entry1/detector/Ny', data = np.float64(np.shape(dicth5['entry1/data/data'])[0]))

def fn_createMaskH5(filename, filenamesave, data_mask):
# to save temp_mask.h5 for the temp.h5 file
    [subs, val, dicth5] = fn_Jonsh5structure()
    filenamesave = filename[0:len(filename)-3] + '_mask.h5'

    f2 = h5py.File(filename, 'r')
    sampleX = f2['entry1/sample/sampleX'];
    sampleY = f2['entry1/sample/sampleY'];
    sampleZ = f2['entry1/sample/sampleZ'];
    depth = f2['entry1/depth'];
    temp_data = f2['entry1/data/data'][:,:]
    for tt in range(len(subs)):
        dicth5[subs[tt]] = f2[subs[tt]]
    # for now data_mask is just np.zeros of the data shape
    #data_mask = np.zeros(np.shape(temp_data));
    #data_mask[temp_data>100] = 255 
    #data_mask[1000:-1,1000:-1] = 255
    dicth5['entry1/data/data'] = data_mask;    

    subs = list(dicth5.keys())
    val = list(dicth5.values())
    with h5py.File(filenamesave, 'w') as f:
        f.attrs.create('file_name', filenamesave)
        f.attrs.create('file_time', '2021-11-13')
        for kk in range(len(subs)):
            f.create_dataset(subs[kk], data=val[kk])
        f.create_dataset('entry1/timingClock', data = np.float64(1))
        f.create_dataset('entry1/depth', data = np.float64(depth))
        f.create_dataset('entry1/timingSignal', data = np.float64(1))
        f['entry1/sample/incident_energy'].attrs.create('units', 'keV')
        f['entry1/depth'].attrs.create('units', 'micron')
        f['entry1/sample/distance'].attrs.create('units', 'mm')
        f['entry1/sample/sampleX'].attrs.create('units', 'micron')
        f['entry1/sample/sampleY'].attrs.create('units', 'micron')
        f['entry1/sample/sampleZ'].attrs.create('units', 'micron')
    f2.close()    
    #print(filenamesave + ' saved')
    

def fn_peakSearch(path_peaksearch, ps_boxsize, ps_maxRfactor, ps_min_size, ps_minSeparation, ps_smooth, ps_threshold, ps_max_peaks, ps_mask, ps_input, ps_out):
    #!/data34/JZT/server336/bin/peaksearch

#USAGE:  peaksearch [-b boxsize -R maxRfactor -m min_size -M max_peaks -s minSeparation -t threshold -T thresholdRatio -p (L,G) -S -K maskFile -D distortionMap] InputImagefileName  OutputPeaksFileName
#switches are:
#	-b box size (half width)
#	-R maximum R factor
#	-m min size of peak (pixels)
#	-M max number of peaks to examine(default=50)
#	-s minimum separation between two peaks (default=2*boxsize)
#	-t user supplied threshold (optional, overrides -T)
#	-T threshold ratio, set threshold to (ratio*[std dev] + avg) (optional)
#	-p use -p L for Lorentzian (default), -p G for Gaussian
#	-S smooth the image
#	-K mask_file_name (use pixels with mask==0)
#	-D distortion map file name
    if len(ps_mask)>0:
        if ps_smooth == 1:
            sub.run([
                    path_peaksearch,
                    '-b', str(ps_boxsize),
                    '-R', str(ps_maxRfactor),
                    '-m', str(ps_min_size),
                    '-s', str(ps_minSeparation),
                    '-t', str(ps_threshold),
                    '-p', 'Gaussian',
                    '-S', f'',
                    '-M', str(ps_max_peaks),
                    '-K', ps_mask,
                    ps_input,
                    ps_out
                ])
        else:
            sub.run([
                    path_peaksearch,
                    '-b', str(ps_boxsize),
                    '-R', str(ps_maxRfactor),
                    '-m', str(ps_min_size),
                    '-s', str(ps_minSeparation),
                    '-t', str(ps_threshold),
                    '-p', 'Gaussian',
                    '-M', str(ps_max_peaks),
                    '-K', ps_mask,
                    ps_input,
                    ps_out
                ])
            
    else:
        if ps_smooth == 1:
            sub.run([
                    path_peaksearch,
                    '-b', str(ps_boxsize),
                    '-R', str(ps_maxRfactor),
                    '-m', str(ps_min_size),
                    '-s', str(ps_minSeparation),
                    '-t', str(ps_threshold),
                    '-p', 'Gaussian',
                    '-S', f'',
                    '-M', str(ps_max_peaks),
                    ps_input,
                    ps_out
                ])
        else:
            sub.run([
                    path_peaksearch,
                    '-b', str(ps_boxsize),
                    '-R', str(ps_maxRfactor),
                    '-m', str(ps_min_size),
                    '-s', str(ps_minSeparation),
                    '-t', str(ps_threshold),
                    '-p', 'Gaussian',
                    '-M', str(ps_max_peaks),
                    ps_input,
                    ps_out
                ])
    #print(ps_out + ' saved')

# extract the peaks from the xml file
def fn_readXY(filename):
    with open(filename, encoding='windows-1252', errors='ignore') as f:
        lines = f.readlines()
       # print(lines)

    for ii in range(len(lines)):
        if lines[ii][0:9] == '$peakList':
            indN = ii

    fitX = np.zeros((len(lines)-indN-1))
    fitY = np.zeros((len(lines)-indN-1))
    Ipeak = np.zeros((len(lines)-indN-1))
    for ii in range(len(lines)-indN-1):
        str1 = lines[indN+1+ii]
        x = str1.split()
        fitX[ii] = np.float64(x[0])
        fitY[ii] = np.float64(x[1])
        Ipeak[ii] = np.float64(x[2])
    return fitX, fitY, Ipeak

def fn_readXYall(filename):
    with open(filename, encoding='windows-1252', errors='ignore') as f:
        lines = f.readlines()
       # print(lines)

    for ii in range(len(lines)):
        if lines[ii][0:9] == '$peakList':
            indN = ii

    fitX = np.zeros((len(lines)-indN-1))
    fitY = np.zeros((len(lines)-indN-1))
    Ipeak = np.zeros((len(lines)-indN-1))
    Integral = np.zeros((len(lines)-indN-1))
    hwhmX = np.zeros((len(lines)-indN-1))
    hwhmY = np.zeros((len(lines)-indN-1))
    tilt = np.zeros((len(lines)-indN-1))
    chisq = np.zeros((len(lines)-indN-1))
    
    for ii in range(len(lines)-indN-1):
        str1 = lines[indN+1+ii]
        x = str1.split()
        fitX[ii] = np.float64(x[0])
        fitY[ii] = np.float64(x[1])
        Ipeak[ii] = np.float64(x[2])
        Integral[ii] = np.float64(x[3])
        hwhmX[ii] = np.float64(x[4])
        hwhmY[ii] = np.float64(x[5])
        tilt[ii] = np.float64(x[6])
        chisq[ii] = np.float64(x[7])
        
    return fitX, fitY, Ipeak, Integral, hwhmX, hwhmY, tilt, chisq

def fn_peaks2qs(path_peaks2qs, pq_geo, pq_cryst, pq_input, pq_out):
# converting the peaks in XY detector to the G unit vectors
# converting the peaks in XY detector to the G unit vectors
#!/data34/JZT/server336/bin/pixels2qs
# usage below
#pixels2qs [-g geometryFile] -x crystalDescriptionFile input_peaks_file output_qs_file
#	input_peaks_file    result from peak search
#	output_qs_file      name for new output file that holds the result
#	switches are:
#		-g geometry file (defaults to geometry.txt)
#		-x crystal description file

#!/data34/JZT/server336/bin/pixels2qs -g "./KaySong/geoN_2021-10-26_18-28-16.xml" -x "./KaySong/Fe.xml" "temp.txt" "temp_Peaks2G.txt"
    sub.run([
                path_peaks2qs,
                '-g', pq_geo,
                '-x', pq_cryst,
                pq_input,
                pq_out
            ])
    print(pq_out + ' saved')


def fn_readPeaks2qs(filename):

    # extract the peaks from the xml file

    with open(filename,encoding='windows-1252', errors='ignore') as f:
        lines = f.readlines()
       # print(lines)

    for ii in range(len(lines)):
        if lines[ii][0:9] == '$N_Ghat+I':
            indN2 = ii

    G_x = np.zeros((len(lines)-indN2-1))
    G_y = np.zeros((len(lines)-indN2-1))
    G_z = np.zeros((len(lines)-indN2-1))
    for ii in range(len(lines)-indN2-1):
        str1 = lines[indN2+1+ii]
        #print(str1)
        x = str1.split(',')
        G_x[ii] = np.float64(x[0])
        G_y[ii] = np.float64(x[1])
        G_z[ii] = np.float64(x[2])
    return G_x, G_y, G_z


def fn_Index(path_index, ix_keVmaxCalc, ix_keVmaxTest, ix_angleTolerance, ix_h, ix_k, ix_l, ix_cone, ix_input, ix_out):

# to run peak search program
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


#!/data34/JZT/server336/bin/euler -q -k 30.0 -t 35.0 -a 0.12 -h 0 0 1 -c 72.0 -f "temp_Peaks2G.txt" -o "temp_4Index.txt"

    sub.run([
                path_index,
                '-q',
                '-k', str(ix_keVmaxCalc),
                '-t', str(ix_keVmaxTest),
                '-a', str(ix_angleTolerance),
                '-c', str(ix_cone),
                '-f', ix_input,
                '-h', str(ix_h), str(ix_k), str(ix_l),
                '-o', ix_out,

            ])
    #print(ix_out + ' saved')
    
    
def fn_readIndex(filename, printoutput):
    
    # extract the peaks from the xml file

    with open(filename,encoding='windows-1252', errors='ignore') as f:
        lines = f.readlines()

    patInd = 0;
    indN3 = [];

    Peaks_calc = []

    for ii in range(len(lines)):
        if lines[ii] == '$pattern'+str(patInd)+' \n':
            patInd = patInd+1;
            indN3.append(ii)

    print('Found '+str(len(indN3))+' patterns')

    for ii in range(len(indN3)):
        Peaks_calc.append([])
        if ii !=len(indN3)-1:
            wr_on = 0;
            for jj in range(indN3[ii+1]-indN3[ii]-1):
                if wr_on == 1:
                    test = lines[indN3[ii]+jj+1];
                    if isinstance(lines[indN3[ii]+jj+1], str):
                        if (lines[indN3[ii]+jj+1]!='\n') and  (len(lines[indN3[ii]+jj+1])!=0):
                            Peaks_calc[ii].append([])
                            s = lines[indN3[ii]+jj+1]
                            s = s.replace("[", "")
                            s = s.replace("]", "")
                            s = s.replace("(", "")
                            s = s.replace(")", "")
                            s = s.replace(",", "")
                            s = s.replace("\n", "")
                            s = re.sub(' +', ' ', s)
                            temp_spl = str.split(s)

                            for kk in range(11):
                              #  print(str(temp_spl[kk]) + ' add')
                                Peaks_calc[ii][appN].append(np.float64(temp_spl[kk]))
                            appN = appN + 1

                if lines[indN3[ii]+jj+1][0:6] == '$array':
                    wr_on = 1;
                    appN = 0;

        else:
            wr_on = 0;
            for jj in range(len(lines)-indN3[ii]-1):
                if wr_on == 1:
                    if isinstance(lines[indN3[ii]+jj+1], str):
                        if (lines[indN3[ii]+jj+1]!='\n') and (len(lines[indN3[ii]+jj+1])!=0):
                            s = lines[indN3[ii]+jj+1]
                            s = s.replace("[", " ")
                            s = s.replace("]", " ")
                            s = s.replace("(", " ")
                            s = s.replace(")", " ")
                            s = s.replace(",", " ")
                            s = s.replace("\n", "")
                            s = re.sub(' +', ' ', s)
                            temp_spl = str.split(s)
                            Peaks_calc[ii].append([])
                            for kk in range(11):
                                Peaks_calc[ii][appN].append(np.float64(temp_spl[kk]))
                            appN = appN + 1
                if lines[indN3[ii]+jj+1][0:6] == '$array':
                    wr_on = 1;
                    appN = 0;

    str_Peaks_calc = ['Peak #', 'G_x', 'G_y', 'G_z', 'G1', 'h', 'k', 'l', 'Inten', 'keV', 'err(deg)', 'PkIndex' ]
    if printoutput == 1:
        for ii in range(len(Peaks_calc)):
            print('Pattern '+ str(ii))
            for jj in range(len(Peaks_calc[ii])):
                print(Peaks_calc[ii][jj])
    return Peaks_calc


def fn_read_abc(file_index):
    Peaks_calc = fn_readIndex(file_index, 0)
    Npatterns = len(Peaks_calc)
    with open(file_index,encoding='windows-1252', errors='ignore') as f:
        lines = f.readlines()
    #print(lines)
    Nind = []
    goodness_index = []
    rms_error_index = []
    astar_all = []
    bstar_all = []
    cstar_all = []
    with open(file_index,encoding='windows-1252', errors='ignore') as f:
        lines = f.readlines()
    # for ii in range(len(lines)):
    #     if lines[ii][0:len('$Nindexed\t\t')] == '$Nindexed\t\t':
    #         s = lines[ii]
    #         s = s.split('\t')
    #         #print(s[2])
    #         Nindexed =  str(s[2])

    
    
    for ii in range(len(lines)):
        for ll in range(len(Peaks_calc)):
            strtemp = '$goodness'+str(ll)
            if lines[ii][0:len(strtemp)] == strtemp:
                s = lines[ii]
                s = s.replace("\n", "")
                s = s.split('\t')
                #print(s)
                goodness_index.append(np.float64(s[2]))
            strtemp = '$rms_error'+str(ll)
            if lines[ii][0:len(strtemp)] == strtemp:
                s = lines[ii]
                s = s.replace("\n", "")
                s = s.split('\t')
                #print(s)
                rms_error_index.append(np.float64(s[2]))
            strtemp = '$recip_lattice'+str(ll)
            if lines[ii][0:len(strtemp)] == strtemp:
                s = lines[ii]
                s = s.replace("\n", "")
                s = s.split('\t')
                #print(s)
                s2 = s[2]
                #print(s[2])
                s2 = s2.split('{')
                #print(s2)
                astar = s2[2]
                bstar = s2[3]
                cstar = s2[4]
                astar = astar.replace("}", "")
                bstar = bstar.replace("}", "")
                cstar = cstar.replace("}", "")
                astar = astar.replace(",", " ")
                bstar = bstar.replace(",", " ")
                cstar = cstar.replace(",", " ")
                #print(astar)
                
                astar_all.append(np.asarray(astar.split(' ')).astype('float64'))
                bstar_all.append(np.asarray(bstar.split(' ')).astype('float64'))
                cstar_all.append(np.asarray(cstar.split(' ')).astype('float64'))
                Nind.append(len(Peaks_calc[ll]))
    return astar_all, bstar_all, cstar_all, goodness_index, rms_error_index, Nind