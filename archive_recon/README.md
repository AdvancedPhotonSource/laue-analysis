If you are reading this... good luck. I'm making this intermediate archive of source code outside of the core package should it need to be referenced in the future. 

These are the archive files of the original wire reconstruction program. As far as I can tell, Jon T. wrote the original code in 2008-2009. Then around 2013-2015, KYUE modified the code adding GPU versions and optimizing the CPU implementations. The original software was maintained in a SVN server which no longer exists. 

Several different file versions, around ~20 in total, were found on in KYUE's old directory and this archive is an attempt to pick that apart for the latest versions of the software at that time. His directory included other artifacts such as testing images and IDE+SVN artifacts. Only the source is committed to this archive on git.

They are split into 2 types, one for reading multiple h5 files from a step scan and another designed to reading a large, single h5 file. In addition to that, GPU and CPU versions of the code exist with some further optimized CPU code versions which optimize memory usage within the core analysis loop of the program.

Here are the original and current file directory mappings in case files need to be pulled from the deep archive again. 

* reconstructMultiple/cpu - /clhome/KYUE/reconstructMultiple/cpucode
* reconstructMultiple/gpu - /clhome/KYUE/reconstructMultiple/DEPTHRECON-M81
* reconstuctSingle/reconstructBigCPU - /clhome/KYUE/reconstructSingle/reconstructBigCPU/reconstructBig
* reconstructSingle/reconstructBigGPU - /clhome/KYUE/reconstructSingle/reconstructBigGPU
* reconstructSingle/cpu_optim - /clhome/KYUE/reconstructSingle/cpucode
* archive_recon/reconstructMultiple/cpu_thread - /data34/Xu/programs/reconstructBP
