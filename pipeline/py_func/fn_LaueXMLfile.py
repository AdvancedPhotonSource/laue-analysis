# import h5py
# import scipy.io as sio
# import csv
# import subprocess as sub
# import re
import datetime as dt
import fn_LaueGopipline2 as LGp
from xml.etree.ElementTree import Element, SubElement, Comment
import numpy as np


def fn_createStepXML (file_peaks, file_p2q, file_index, extraLibStep, extraLibDet, xtal_element):
    generated_on = str(dt.datetime.now())
    # here read the files and extract from there most of the parameters
    
    
    
    step_ = Element('step')
    step_.set('xmlns', 'python3p8')
    title_ = SubElement(step_, 'title')
    sampleName_ = SubElement(step_, 'sampleName')
    date_       = SubElement(step_, 'date')
    beamBad_       = SubElement(step_, 'beamBad')
    CCDshutter_       = SubElement(step_, 'CCDshutter')
    lightOn_       = SubElement(step_, 'lightOn')
    monoMode_       = SubElement(step_, 'monoMode')
    Xsample_       = SubElement(step_, 'Xsample')
    Ysample_       = SubElement(step_, 'Ysample')
    Zsample_       = SubElement(step_, 'Zsample')
    depth_       = SubElement(step_, 'depth')
    energy_       = SubElement(step_, 'energy')
    energy_.set('unit', 'keV')
    hutchTemperature_       = SubElement(step_, 'hutchTemperature')
    sampleDistance_       = SubElement(step_, 'sampleDistance')

    detector_       = Element('detector')
    inputImage_       = SubElement(detector_, 'inputImage')
    detectorID_       = SubElement(detector_, 'detectorID')
    exposure_       = SubElement(detector_, 'exposure')
    exposure_.set('unit', 'sec')
    Nx_       = SubElement(detector_, 'Nx')
    Ny_       = SubElement(detector_, 'Ny')
    totalSum_       = SubElement(detector_, 'totalSum')
    sumAboveThreshold_       = SubElement(detector_, 'sumAboveThreshold')
    numAboveThreshold_       = SubElement(detector_, 'numAboveThreshold')
    cosmicFilter_       = SubElement(detector_, 'cosmicFilter')
    cosmicFilter_.text = 'True'
    geoFile_       = SubElement(detector_, 'geoFile')
    ROI_       = SubElement(detector_, 'ROI')
    ROI_.text = ''
    
    # I'm cheating from here: need to correct here for now
    ROI_.set('startx', '0')
    ROI_.set('endx', '2047')
    ROI_.set('groupx', '1')
    ROI_.set('starty', '0')
    ROI_.set('endy', '2047')
    ROI_.set('groupy', '1')
    
    peaksXY_       = Element('peaksXY')
    peaksXY_.set('peakProgram', 'peaksearch')
    peaksXY_.set('minwidth', '0.275')
    peaksXY_.set('threshold', '20.0')
    peaksXY_.set('thresholdRatio', '-1')
    peaksXY_.set('maxRfactor', '0.8')
    peaksXY_.set('maxwidth', '7.5')
    peaksXY_.set('maxCentToFit', '5.0')
    peaksXY_.set('boxsize', '5')
    peaksXY_.set('max_number', '200')
    peaksXY_.set('min_separation', '20')
    peaksXY_.set('peakShape', 'Lorentzian')
    
    
    
    
    
    
    
    
    [fitX, fitY, Inten, Integ, hwhmX, hwhmY, tilt, chisq] = LGp.fn_readXYall(file_peaks)
    Npeaks = len(fitX)
    ltemp = list(fitX)
    str_Xpixels = ' '.join(str(x) for x in ltemp)
    ltemp = list(fitY)
    str_Ypixels = ' '.join(str(x) for x in ltemp)
    ltemp = list(Inten)
    str_Intens = ' '.join(str(x) for x in ltemp)
    ltemp = list(Integ)
    str_Integral = ' '.join(str(x) for x in ltemp)
    ltemp = list(hwhmX)
    str_hwhmX = ' '.join(str(x) for x in ltemp)
    ltemp = list(hwhmY)
    str_hwhmY = ' '.join(str(x) for x in ltemp)
    ltemp = list(tilt)
    str_tilt = ' '.join(str(x) for x in ltemp)
    ltemp = list(chisq)
    str_chisq = ' '.join(str(x) for x in ltemp)
    
    
    [Qx, Qy, Qz] = LGp.fn_readPeaks2qs(file_p2q)
    ltemp = list(Qx)
    str_Qx = ' '.join(str(x) for x in ltemp)
    ltemp = list(Qy)
    str_Qy = ' '.join(str(x) for x in ltemp)
    ltemp = list(Qz)
    str_Qz = ' '.join(str(x) for x in ltemp)
    
    
    Xpixel_       = SubElement(peaksXY_, 'Xpixel')
    Xpixel_.text = str_Xpixels
    Ypixel_       = SubElement(peaksXY_, 'Ypixel')
    Ypixel_.text = str_Ypixels
    Intens_       = SubElement(peaksXY_, 'Intens')
    Intens_.text = str_Intens
    Integral_       = SubElement(peaksXY_, 'Integral')
    Integral_.text = str_Integral
    hwhmX_       = SubElement(peaksXY_, 'hwhmX')
    hwhmX_.set('unit', 'pixel')
    hwhmX_.text = str_hwhmX
    hwhmY_       = SubElement(peaksXY_, 'hwhmY')
    hwhmY_.set('unit', 'pixel')
    hwhmY_.text = str_hwhmY
    tilt_       = SubElement(peaksXY_, 'tilt')
    tilt_.set('unit', 'degree')
    tilt_.text = str_tilt
    chisq_       = SubElement(peaksXY_, 'chisq')
    chisq_.text = str_chisq
    
    Qx_       = SubElement(peaksXY_, 'Qx')
    Qx_.text = str_Qx
    Qy_       = SubElement(peaksXY_, 'Qy')
    Qy_.text = str_Qy
    Qz_       = SubElement(peaksXY_, 'Qz')
    Qz_.text = str_Qz


    
    peaksXY_.set('Npeaks', str(Npeaks))
    peaksXY_.set('executionTime', '3.18') # cheat here
    
    
    detector_.append(peaksXY_)
    for ii in range(len(detector_)):
        try:
            detector_[ii].text = extraLibDet[detector_[ii].tag]
        except:
            next 
    
    step_.append(detector_)
    
    
    
    #indexing_       = SubElement(step_, 'indexing')
    indexing_       = SubElement(step_, 'indexing')
    
    
    indexing_.set('indexProgram', 'euler')
    indexing_.set('executionTime', '0.07')

    indexing_.append(Comment(' Results of indexing '))
    
    
    Peaks_calc = LGp.fn_readIndex(file_index, 0)
    
    indexing_.set('Npatterns', str(len(Peaks_calc)))
    indexing_.set('Npeaks', str(Npeaks))
    
    goodness_index = []
    rms_error_index = []
    astar_all = []
    bstar_all = []
    cstar_all = []
    with open(file_index,encoding='windows-1252', errors='ignore') as f:
        lines = f.readlines()
        
    for ii in range(len(lines)):
        if lines[ii][0:len('$keVmaxCalc\t\t')] == '$keVmaxCalc\t\t':
            s = lines[ii]
            s = s.split('\t')
            #print(s[2])
            indexing_.set('keVmaxCalc', str(s[2]))
        if lines[ii][0:len('$angleTolerance\t\t')] == '$angleTolerance\t\t':
            s = lines[ii]
            s = s.split('\t')
            #print(s[2])
            indexing_.set('angleTolerance', str(s[2]))
        if lines[ii][0:len('$keVmaxTest\t\t')] == '$keVmaxTest\t\t':
            s = lines[ii]
            s = s.split('\t')
            #print(s[2])
            indexing_.set('keVmaxTest', str(s[2]))
        if lines[ii][0:len('$cone\t\t')] == '$cone\t\t':
            s = lines[ii]
            s = s.split('\t')
            #print(s[2])
            indexing_.set('cone', str(s[2]))
        if lines[ii][0:len('$Nindexed\t\t')] == '$Nindexed\t\t':
            s = lines[ii]
            s = s.split('\t')
            #print(s[2])
            indexing_.set('Nindexed', str(s[2]))
        if lines[ii][0:len('$hklPrefer\t\t')] == '$hklPrefer\t\t':
            s = lines[ii]
            s = s.split('\t')
            s2 = s[2]
            s2 = s2.replace(",", " ")
            s2 = s2.replace("}", "")
            s2 = s2.replace("{", "")
            #print(s2)
            indexing_.set('hklPrefer', str(s2))
        
    for ii in range(len(lines)):
        for ll in range(len(Peaks_calc)):
            strtemp = '$goodness'+str(ll)
            if lines[ii][0:len(strtemp)] == strtemp:
                s = lines[ii]
                s = s.replace("\n", "")
                s = s.split('\t')
                #print(s)
                goodness_index.append(s[2])
            strtemp = '$rms_error'+str(ll)
            if lines[ii][0:len(strtemp)] == strtemp:
                s = lines[ii]
                s = s.replace("\n", "")
                s = s.split('\t')
                #print(s)
                rms_error_index.append(s[2])
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
                
                
                
                
                
                astar_all.append(astar)
                bstar_all.append(bstar)
                cstar_all.append(cstar)
                #print(astar)
                #print(bstar)
                #print(cstar)
                
    for ii in range(len(Peaks_calc)):
        pattern_       = Element('pattern')
        pattern_.set('num', str(ii))
        pattern_.set('rms_error', rms_error_index[ii])
        pattern_.set('goodness', goodness_index[ii])
        pattern_.set('Nindexed', str(len(Peaks_calc[ii])))


        recip_lattice_       = SubElement(pattern_, 'recip_lattice')
        recip_lattice_.set('unit', '1/nm')
        astar_       = SubElement(recip_lattice_, 'astar')
        astar_.text = str(astar_all[ii])
        bstar_       = SubElement(recip_lattice_, 'bstar')
        bstar_.text = str(bstar_all[ii])
        cstar_       = SubElement(recip_lattice_, 'cstar')
        cstar_.text = str(cstar_all[ii])

        hkl_s_       = SubElement(pattern_, 'hkl_s')
        
        
        temp_arr = np.asarray(Peaks_calc[ii])
        ltemp = list(temp_arr[:,4])
        str_h = ' '.join(str(int(x)) for x in ltemp)
        
        temp_arr = np.asarray(Peaks_calc[ii])
        ltemp = list(temp_arr[:,5])
        str_k = ' '.join(str(int(x)) for x in ltemp)
        
        temp_arr = np.asarray(Peaks_calc[ii])
        ltemp = list(temp_arr[:,6])
        str_l = ' '.join(str(int(x)) for x in ltemp)
        
        temp_arr = np.asarray(Peaks_calc[ii])
        ltemp = list(temp_arr[:,10])
        str_peakind = ' '.join(str(int(x)) for x in ltemp)

        h_       = SubElement(hkl_s_, 'h')
        h_.text = str_h
        k_       = SubElement(hkl_s_, 'k')
        k_.text = str_k
        l_       = SubElement(hkl_s_, 'l')
        l_.text = str_l
        PkIndex_       = SubElement(hkl_s_, 'PkIndex')
        PkIndex_.text = str_peakind
        
        indexing_.append(pattern_)
    
    
    
    indexing_.append(xtal_element)
    
    
    
    
    
    
    
    for ii in range(len(step_)):
        try:
            step_[ii].text = extraLibStep[step_[ii].tag]
        except:
            next
    
    return step_


def fn_createStepXML_sym (file_peaks, file_p2q, file_index, extraLibStep, extraLibDet, xtal_element, SymMat, aref, bref, cref, deltaQ):
    generated_on = str(dt.datetime.now())
    # here read the files and extract from there most of the parameters
    
    sym_failed = []
    
    step_ = Element('step')
    step_.set('xmlns', 'python3p8')
    title_ = SubElement(step_, 'title')
    sampleName_ = SubElement(step_, 'sampleName')
    date_       = SubElement(step_, 'date')
    beamBad_       = SubElement(step_, 'beamBad')
    CCDshutter_       = SubElement(step_, 'CCDshutter')
    lightOn_       = SubElement(step_, 'lightOn')
    monoMode_       = SubElement(step_, 'monoMode')
    Xsample_       = SubElement(step_, 'Xsample')
    Ysample_       = SubElement(step_, 'Ysample')
    Zsample_       = SubElement(step_, 'Zsample')
    depth_       = SubElement(step_, 'depth')
    energy_       = SubElement(step_, 'energy')
    energy_.set('unit', 'keV')
    hutchTemperature_       = SubElement(step_, 'hutchTemperature')
    sampleDistance_       = SubElement(step_, 'sampleDistance')

    detector_       = Element('detector')
    inputImage_       = SubElement(detector_, 'inputImage')
    detectorID_       = SubElement(detector_, 'detectorID')
    exposure_       = SubElement(detector_, 'exposure')
    exposure_.set('unit', 'sec')
    Nx_       = SubElement(detector_, 'Nx')
    Ny_       = SubElement(detector_, 'Ny')
    totalSum_       = SubElement(detector_, 'totalSum')
    sumAboveThreshold_       = SubElement(detector_, 'sumAboveThreshold')
    numAboveThreshold_       = SubElement(detector_, 'numAboveThreshold')
    cosmicFilter_       = SubElement(detector_, 'cosmicFilter')
    cosmicFilter_.text = 'True'
    geoFile_       = SubElement(detector_, 'geoFile')
    ROI_       = SubElement(detector_, 'ROI')
    ROI_.text = ''
    
    # I'm cheating from here: need to correct here for now
    ROI_.set('startx', '0')
    ROI_.set('endx', '2047')
    ROI_.set('groupx', '1')
    ROI_.set('starty', '0')
    ROI_.set('endy', '2047')
    ROI_.set('groupy', '1')
    
    peaksXY_       = Element('peaksXY')
    peaksXY_.set('peakProgram', 'peaksearch')
    peaksXY_.set('minwidth', '0.275')
    peaksXY_.set('threshold', '20.0')
    peaksXY_.set('thresholdRatio', '-1')
    peaksXY_.set('maxRfactor', '0.8')
    peaksXY_.set('maxwidth', '7.5')
    peaksXY_.set('maxCentToFit', '5.0')
    peaksXY_.set('boxsize', '5')
    peaksXY_.set('max_number', '200')
    peaksXY_.set('min_separation', '20')
    peaksXY_.set('peakShape', 'Lorentzian')
    
    
    
    
    
    
    
    
    [fitX, fitY, Inten, Integ, hwhmX, hwhmY, tilt, chisq] = LGp.fn_readXYall(file_peaks)
    Npeaks = len(fitX)
    ltemp = list(fitX)
    str_Xpixels = ' '.join(str(x) for x in ltemp)
    ltemp = list(fitY)
    str_Ypixels = ' '.join(str(x) for x in ltemp)
    ltemp = list(Inten)
    str_Intens = ' '.join(str(x) for x in ltemp)
    ltemp = list(Integ)
    str_Integral = ' '.join(str(x) for x in ltemp)
    ltemp = list(hwhmX)
    str_hwhmX = ' '.join(str(x) for x in ltemp)
    ltemp = list(hwhmY)
    str_hwhmY = ' '.join(str(x) for x in ltemp)
    ltemp = list(tilt)
    str_tilt = ' '.join(str(x) for x in ltemp)
    ltemp = list(chisq)
    str_chisq = ' '.join(str(x) for x in ltemp)
    
    
    [Qx, Qy, Qz] = LGp.fn_readPeaks2qs(file_p2q)
    ltemp = list(Qx)
    str_Qx = ' '.join(str(x) for x in ltemp)
    ltemp = list(Qy)
    str_Qy = ' '.join(str(x) for x in ltemp)
    ltemp = list(Qz)
    str_Qz = ' '.join(str(x) for x in ltemp)
    
    
    Xpixel_       = SubElement(peaksXY_, 'Xpixel')
    Xpixel_.text = str_Xpixels
    Ypixel_       = SubElement(peaksXY_, 'Ypixel')
    Ypixel_.text = str_Ypixels
    Intens_       = SubElement(peaksXY_, 'Intens')
    Intens_.text = str_Intens
    Integral_       = SubElement(peaksXY_, 'Integral')
    Integral_.text = str_Integral
    hwhmX_       = SubElement(peaksXY_, 'hwhmX')
    hwhmX_.set('unit', 'pixel')
    hwhmX_.text = str_hwhmX
    hwhmY_       = SubElement(peaksXY_, 'hwhmY')
    hwhmY_.set('unit', 'pixel')
    hwhmY_.text = str_hwhmY
    tilt_       = SubElement(peaksXY_, 'tilt')
    tilt_.set('unit', 'degree')
    tilt_.text = str_tilt
    chisq_       = SubElement(peaksXY_, 'chisq')
    chisq_.text = str_chisq
    
    Qx_       = SubElement(peaksXY_, 'Qx')
    Qx_.text = str_Qx
    Qy_       = SubElement(peaksXY_, 'Qy')
    Qy_.text = str_Qy
    Qz_       = SubElement(peaksXY_, 'Qz')
    Qz_.text = str_Qz


    
    peaksXY_.set('Npeaks', str(Npeaks))
    peaksXY_.set('executionTime', '3.18') # cheat here
    
    
    detector_.append(peaksXY_)
    for ii in range(len(detector_)):
        try:
            detector_[ii].text = extraLibDet[detector_[ii].tag]
        except:
            next 
    
    step_.append(detector_)
    
    
    
    #indexing_       = SubElement(step_, 'indexing')
    indexing_       = SubElement(step_, 'indexing')
    
    
    indexing_.set('indexProgram', 'euler')
    indexing_.set('executionTime', '0.07')

    indexing_.append(Comment(' Results of indexing '))
    
    
    Peaks_calc = LGp.fn_readIndex(file_index, 0)
    
    indexing_.set('Npatterns', str(len(Peaks_calc)))
    indexing_.set('Npeaks', str(Npeaks))
    
    goodness_index = []
    rms_error_index = []
    astar_all = []
    bstar_all = []
    cstar_all = []
    with open(file_index,encoding='windows-1252', errors='ignore') as f:
        lines = f.readlines()
        
    for ii in range(len(lines)):
        if lines[ii][0:len('$keVmaxCalc\t\t')] == '$keVmaxCalc\t\t':
            s = lines[ii]
            s = s.split('\t')
            #print(s[2])
            indexing_.set('keVmaxCalc', str(s[2]))
        if lines[ii][0:len('$angleTolerance\t\t')] == '$angleTolerance\t\t':
            s = lines[ii]
            s = s.split('\t')
            #print(s[2])
            indexing_.set('angleTolerance', str(s[2]))
        if lines[ii][0:len('$keVmaxTest\t\t')] == '$keVmaxTest\t\t':
            s = lines[ii]
            s = s.split('\t')
            #print(s[2])
            indexing_.set('keVmaxTest', str(s[2]))
        if lines[ii][0:len('$cone\t\t')] == '$cone\t\t':
            s = lines[ii]
            s = s.split('\t')
            #print(s[2])
            indexing_.set('cone', str(s[2]))
        if lines[ii][0:len('$Nindexed\t\t')] == '$Nindexed\t\t':
            s = lines[ii]
            s = s.split('\t')
            #print(s[2])
            indexing_.set('Nindexed', str(s[2]))
        if lines[ii][0:len('$hklPrefer\t\t')] == '$hklPrefer\t\t':
            s = lines[ii]
            s = s.split('\t')
            s2 = s[2]
            s2 = s2.replace(",", " ")
            s2 = s2.replace("}", "")
            s2 = s2.replace("{", "")
            #print(s2)
            indexing_.set('hklPrefer', str(s2))
        
    for ii in range(len(lines)):
        for ll in range(len(Peaks_calc)):
            strtemp = '$goodness'+str(ll)
            if lines[ii][0:len(strtemp)] == strtemp:
                s = lines[ii]
                s = s.replace("\n", "")
                s = s.split('\t')
                #print(s)
                goodness_index.append(s[2])
            strtemp = '$rms_error'+str(ll)
            if lines[ii][0:len(strtemp)] == strtemp:
                s = lines[ii]
                s = s.replace("\n", "")
                s = s.split('\t')
                #print(s)
                rms_error_index.append(s[2])
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
                
                a_ll = np.asarray(astar.split(' ')).astype('float64')
                b_ll = np.asarray(bstar.split(' ')).astype('float64')
                c_ll = np.asarray(cstar.split(' ')).astype('float64')
                
                temp = np.zeros((3,3))
                temp[:,0] = a_ll 
                temp[:,1] = b_ll
                temp[:,2] = c_ll
                
                sym_key = 0
                for ss in range(len(SymMat)):
                    abc1 = temp@SymMat[ss]

                    a1 = abc1[:,0]
                    b1 = abc1[:,1]
                    c1 = abc1[:,2]


                    if np.linalg.norm(a1 - aref) + np.linalg.norm(b1-bref) + np.linalg.norm(c1-cref) <= deltaQ: 
                        sym_key = 1
                        astar = str(a1[0]) + ' ' + str(a1[1]) + ' ' + str(a1[2]) 
                        bstar = str(b1[0]) + ' ' + str(b1[1]) + ' ' + str(b1[2]) 
                        cstar = str(c1[0]) + ' ' + str(c1[1]) + ' ' + str(c1[2]) 
                if sym_key == 0:
                    sym_failed.append(str(ll) + ' filename: ' + file_peaks)
                
                
                astar_all.append(astar)
                bstar_all.append(bstar)
                cstar_all.append(cstar)
                #print(astar)
                #print(bstar)
                #print(cstar)
                
    for ii in range(len(Peaks_calc)):
        pattern_       = Element('pattern')
        pattern_.set('num', str(ii))
        pattern_.set('rms_error', rms_error_index[ii])
        pattern_.set('goodness', goodness_index[ii])
        pattern_.set('Nindexed', str(len(Peaks_calc[ii])))


        recip_lattice_       = SubElement(pattern_, 'recip_lattice')
        recip_lattice_.set('unit', '1/nm')
        astar_       = SubElement(recip_lattice_, 'astar')
        astar_.text = str(astar_all[ii])
        bstar_       = SubElement(recip_lattice_, 'bstar')
        bstar_.text = str(bstar_all[ii])
        cstar_       = SubElement(recip_lattice_, 'cstar')
        cstar_.text = str(cstar_all[ii])

        hkl_s_       = SubElement(pattern_, 'hkl_s')
        
        
        temp_arr = np.asarray(Peaks_calc[ii])
        ltemp = list(temp_arr[:,4])
        str_h = ' '.join(str(int(x)) for x in ltemp)
        
        temp_arr = np.asarray(Peaks_calc[ii])
        ltemp = list(temp_arr[:,5])
        str_k = ' '.join(str(int(x)) for x in ltemp)
        
        temp_arr = np.asarray(Peaks_calc[ii])
        ltemp = list(temp_arr[:,6])
        str_l = ' '.join(str(int(x)) for x in ltemp)
        
        temp_arr = np.asarray(Peaks_calc[ii])
        ltemp = list(temp_arr[:,10])
        str_peakind = ' '.join(str(int(x)) for x in ltemp)

        h_       = SubElement(hkl_s_, 'h')
        h_.text = str_h
        k_       = SubElement(hkl_s_, 'k')
        k_.text = str_k
        l_       = SubElement(hkl_s_, 'l')
        l_.text = str_l
        PkIndex_       = SubElement(hkl_s_, 'PkIndex')
        PkIndex_.text = str_peakind
        
        indexing_.append(pattern_)
    
    
    
    indexing_.append(xtal_element)
    
    
    
    
    
    
    
    for ii in range(len(step_)):
        try:
            step_[ii].text = extraLibStep[step_[ii].tag]
        except:
            next
    
    return step_, sym_failed




