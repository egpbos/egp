"""
MMF_io.py
Reduced version of MMF.py module from MMF/NEXUS+

Authors: Marius Cautun (creator), E. G. Patrick Bos (modifications + reduction).
Copyright (c) 2012-2018. All rights reserved.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os.path
from miscellaneous import throwError, charToString, readArrayEntries, writeArrayEntries, sourceDirectoryPath, checkArraySizes
import analysis

currentDirectory = sourceDirectoryPath
bufferType = np.uint64
bufferSizeBytes = 8


# the different types of MMF environments
MMF_NODE = 4
MMF_FILAMENT = 3
MMF_WALL = 2
MMF_ALL = 5


# the different types of MMF response output
MMF_RESPONSE = 1           # contains the response value at a given scale
MMF_EIGEN = 5              # contains the eigenvalues and eigenvectors for a given scale
MMF_EIGENVECTOR = 6        # contains the significant eigenvectors for a given features (eigenvector 3 for filaments -the direction along the filament- and eigenvector 1 for wall -the direction perpendicular to the wall-)
MMF_MAX_RESPONSE = 10      # contains the values of the maximum MMF response
MMF_MAX_RESPONSE_SCALE = 11    # contains the values of the maximum MMF response and also the scale corresponding to the maximum response
MMF_MAX_EIGEN = 15         # the eigenvalues and eigenvectors corresponding to the maximum response scale
MMF_MAX_EIGENVECTOR = 16   # same as 6, but corresponding to the maximum response scale
MMF_CLEAN_RESPONSE = 20    # contains a short int data with values of 0 or 1, depending if the pixel is a valid feature or not (e.g. for filaments: 1=pixel is part of a filament, 0=pixel is not part of the filament)
MMF_CLEAN_RESPONSE_COMBINED = 21 # contains the above information, but for all the environments: 0 = void, 2 = wall, 3 = filament and 4 = node
MMF_OBJECTS = 30           # all the pixels corresponding to the same object have the same value, the object tag/id
MMF_DIRECTIONS = 40        # contains the directions for the valid fialment/wall cells
MMF_PROPERTIES = 50        # contains the properties (thickness and mass density) for the valid fialment/wall cells


MMFFeature = { 4:'node', 3:'filament', 2:'wall', -1:'unknown', 5:'all environments' }
MMFFilterTypes = { 40:'node filter 40 (from Miguel thesis)', 41:'node filter 41', 42:'node filter 42', 43:'node filter 43', 30:'filament filter 30 (from Miguel thesis)', 31:'filament filter 31', 32:'filament filter 32', 33:'filament filter 33', 20:'wall filter 20 (from Miguel thesis)', 21:'wall filter 21', 22:'wall filter 22', 23:'wall filter 23', -1:'Unknown', 50:'multiple filters since combined file' }
MMFFileType = { 1:"response", 5:"response eigenvalues & eigenvectors", 6:"response eigenvectors giving direction for filaments and walls", 10:"maximum response", 11:"maximum response & scale", 15:"maximum response eigenvalues & eigenvectors", 16:"maximum response eigenvectors giving direction for filaments and walls", 20:"MMF clean response", 21:"MMF combined clean response", 30:"MMF objects", 40:"filament/wall directions", 50:"filament/wall properties (thickness and mass density)",  -1:"Unknown" }
MMFVariableName = { 1:"response", 5:"eigenvaluesEigenvectors", 6:"environmentEigenvectors", 10:"maxResponse", 11:"maxResponse", 15:"maxEigenvaluesEigenvectors", 16:"maxEnvironmentEigenvectors", 20:"cleanResponse", 21:"combinedCleanResponse", 30:"MMFObjects", 40:"MMFDirections", 50:"MMFProperties", -1:"unknown" }
MMFDataType = { 1:'f4', 5:'f4', 6:'f4', 10:'f4', 11:'f4', 15:'f4', 16:'f4', 20:'i2', 21:'i2', 30:'i4', 40:'f4', 50:'f4', -1:'f4'}
MMFDataComponents = { 1:1, 2:3, 5:9, 6:3, 10:1, 11:3, 15:9, 16:3, 20:1, 21:1, 30:1, 40:1, 50:1, -1:1 }
MMFMethod = {1:'density', 100:'logarithm filtering of density', 5:'density logarithm', 10:'gravitational potential', 20:'velocity divergence', 25:'velocity divergence logarithm', 30:'velocity potential', -1:'Unknown' }


class MMFHeader:
    """ A class used for reading and storing the header of a binary MMF grid file. It uses the numpy class to define the variables. """
    byteSize = 1024
    fillSize = byteSize- 16*8- 18*8 - 8
    
    def __init__(self):
        #variables related to the density computation
        self.gridSize = np.zeros( 3, dtype=np.uint64 )
        self.totalGrid = np.uint64( 0 )
        self.feature = np.int32( -10 )
        self.scale = np.int32( -10 )
        self.radius = np.float32( -1. )
        self.bias = np.float32( -1. )
        self.filter = np.int32( -1 )
        self.fileType = np.int32( -1 )
        self.noMMFFiles = np.int32( 1 )
        self.MMFFileGrid = np.zeros( 3, dtype=np.int32 )
        self.indexMMFFile = np.int32( -1 )
        self.method = np.int32( -1 )
        self.box = np.zeros( 6, dtype=np.float64 )
        
        #variables from the Gadget snapshot
        self.npartTotal = np.zeros( 6, dtype=np.uint64 )
        self.mass = np.zeros( 6, dtype=np.float64 )
        self.time = np.float64( 0. )
        self.redshift = np.float64( 0. )
        self.BoxSize = np.float64( 0. )
        self.Omega0 = np.float64( 0. )
        self.OmegaLambda = np.float64( 0. )
        self.HubbleParam = np.float64( 0. )
        
        #additional information about files
        self.fill = np.zeros( MMFHeader.fillSize, dtype='c' )
        self.FILE_ID = np.int64( 10 )
    
    def SetType(self):
        self.totalGrid = np.uint64( self.totalGrid )
        self.feature = np.int32( self.feature )
        self.scale = np.int32( self.scale )
        self.radius = np.float32( self.radius )
        self.bias = np.float32( self.bias )
        self.filter = np.int32( self.filter )
        self.fileType = np.int32( self.fileType )
        self.noMMFFiles = np.int32( self.noMMFFiles )
        self.indexMMFFile = np.int32( self.indexMMFFile )
        self.method = np.int32( self.method )
        self.time = np.float64( self.time )
        self.redshift = np.float64( self.redshift )
        self.BoxSize = np.float64( self.BoxSize )
        self.Omega0 = np.float64( self.Omega0 )
        self.OmegaLambda = np.float64( self.OmegaLambda )
        self.HubbleParam = np.float64(self.HubbleParam  )
        self.FILE_ID = np.int64( self.FILE_ID )
    
    def nbytes(self):
        __size =  self.gridSize.nbytes + self.totalGrid.nbytes + self.feature.nbytes + self.scale.nbytes + self.radius.nbytes + self.bias.nbytes + self.filter.nbytes + self.fileType.nbytes + self.noMMFFiles.nbytes + self.MMFFileGrid.nbytes + self.indexMMFFile.nbytes + self.method.nbytes + self.box.nbytes + self.npartTotal.nbytes + self.mass.nbytes + self.time.nbytes + self.redshift.nbytes + self.BoxSize.nbytes + self.Omega0.nbytes + self.OmegaLambda.nbytes + self.HubbleParam.nbytes + self.fill.nbytes + self.FILE_ID.nbytes
        return __size
    
    def dtype(self):
        __dt = np.dtype([ ('gridSize',np.uint64,3), ('totalGrid',np.uint64), ('feature',np.int32), ('scale',np.int32), ('radius',np.float32), ('bias',np.float32), ('filter',np.int32), ('fileType',np.int32), ('noMMFFiles',np.int32), ('MMFFileGrid',np.int32,3), ('indexMMFFile',np.int32), ('method',np.int32), ('box',np.float64,6), ('npartTotal',np.uint64,6), ('mass',np.float64,6), ('time',np.float64), ('redshift',np.float64), ('BoxSize',np.float64), ('Omega0',np.float64), ('OmegaLambda',np.float64), ('HubbleParam',np.float64), ('fill','c',MMFHeader.fillSize), ('FILE_ID',np.int64) ])
        return __dt
    
    def TupleAsString(self):
        return "( self.gridSize, self.totalGrid, self.feature, self.scale, self.radius, self.bias, self.filter, self.fileType, self.noMMFFiles, self.MMFFileGrid, self.indexMMFFile, self.method, self.box, self.npartTotal, self.mass, self.time, self.redshift, self.BoxSize, self.Omega0, self.OmegaLambda, self.HubbleParam, self.fill, self.FILE_ID )"
    
    def Tuple(self):
        return eval(self.TupleAsString())
    
    def fromfile(self,f,BUFFER=True):
        if BUFFER: __buffer1 = np.fromfile( f, bufferType, 1 )[0]
        A = np.fromfile( f, self.dtype(), 1)[0]
        if BUFFER:
            __buffer2 = np.fromfile( f, bufferType, 1 )[0]
            if ( __buffer1!=__buffer2 or __buffer1!=MMFHeader.byteSize ):
                throwError( "Error reading the header of the MMF file. 'buffer1'=", __buffer1, "and 'buffer2'=", __buffer2, "when both should be", MMFHeader.byteSize )
        exec( "%s = A" % self.TupleAsString() )
    
    def tofile(self,f,BUFFER=True):
        self.SetType()
        if self.nbytes()!=MMFHeader.byteSize:
            throwError("When calling the function 'MMFHeader.tofile()'. The size of the MMF header is %i while the expected size is %i. Please check which variables do not have the correct size." % (self.nbytes(),MMFHeader.byteSize))
        __buffer = np.array( [self.nbytes()], dtype=np.uint64 )
        __A = np.array( [self.Tuple()], dtype=self.dtype() )
        if BUFFER: __buffer.tofile( f )
        __A.tofile( f )
        if BUFFER: __buffer.tofile( f )
    
    def Feature(self):
        __feature = 'Unknown'
        if self.feature in MMFFeature: __feature = MMFFeature[ self.feature ]
        return ( self.feature, __feature )
    
    def FilterType(self):
        __filterType = 'Unknown'
        if self.filter in MMFFilterTypes: __filterType = MMFFilterTypes[ self.filter ]
        return ( self.filter, __filterType )
    
    def FileType(self):
        """ Returns the type of data stored in the file. """
        __fileType = 'Unknown'
        if self.fileType in MMFFileType: __fileType = MMFFileType[ self.fileType ]
        return ( self.fileType, __fileType )
    
    def DataType(self):
        """ Returns the type of the data in the file, NOT the type of the header. """
        __dataType = 'f4'
        if self.fileType in MMFDataType: __dataType = MMFDataType[ self.fileType ]
        return __dataType
    
    def DataComponents(self):
        """ Returns the  number of components of the data in the file. """
        __noComponents = 1
        if self.fileType in MMFDataComponents: __noComponents = MMFDataComponents[ self.fileType ]
        return __noComponents
    
    def DataName(self):
        """ Returns the name of the variable stored in the file. """
        __dataName = "unknown"
        if self.fileType in MMFVariableName: __dataName = MMFVariableName[ self.fileType ]
        return __dataName
    
    def Method(self):
        """ Returns the method used to compute the MMF data. """
        __method = 'Unknown'
        if self.method in MMFMethod: __method = MMFMethod[ self.method ]
        return ( self.method, __method )
    
    def BoxLength(self):
        __boxLength = np.zeros( self.box.size // 2, self.box.dtype )
        __boxLength[:] = self.box[1::2] - self.box[0::2]
        return __boxLength
    
    def PrintValues(self):
        print("The values contained in the MMF header:")
        print("1) Information about the file itself:")
        print("  gridSize        = ", self.gridSize)
        print("  totalGrid       = ", self.totalGrid)
        print("  feature         = ", self.Feature())
        print("  scale           = ", self.scale)
        print("  radius          = ", self.radius)
        print("  bias            = ", self.bias)
        print("  filter          = ", self.FilterType())
        print("  fileType        = ", self.FileType())
        print("  noMMFFiles      = ", self.noMMFFiles)
        if self.noMMFFiles>1 :
            print("  MMFFileGrid     = ", self.MMFFileGrid)
            print("  indexMMFFile    = ", self.indexMMFFile)
        print("  method          = ", self.Method())
        print("  box coordinates = ", self.box)
        
        print("\n2) Information about the Gadget snapshot used to compute the density:")
        print("  npartTotal   = ", self.npartTotal)
        print("  mass         = ", self.mass)
        print("  time         = ", self.time)
        print("  redshift     = ", self.redshift)
        print("  BoxSize      = ", self.BoxSize)
        print("  Omega0       = ", self.Omega0)
        print("  OmegaLambda  = ", self.OmegaLambda)
        print("  HubbleParam  = ", self.HubbleParam)
        
        print("\n3) Additional information:")
        print("  fill         = %s" % charToString(self.fill))
        print()
    
    def AddProgramCommands(self,commands):
        """Adds the program options used to obtain the current results to the 'fill' array in the header."""
        newCommands = self.fill.tostring().rstrip('\x00') + commands + ' ;  '
        choice = int( len(newCommands) < MMFHeader.fillSize )
        newLen = [ MMFHeader.fillSize, len(newCommands) ] [ choice ]
        newOff = [ len(newCommands)-MMFHeader.fillSize, 0 ] [ choice ]
        self.fill[:newLen] = newCommands[newOff:newLen]



def MMFMultipleFiles(rootName,fileIndex):
    """ Returns the name of the MMF file 'fileIndex' when a result is saved in multiple binary files.
    It takes 2 arguments: root name of the files and the file number whose name is requested (from 0 to MMFHeader.noDensityFiles-1)."""
    return rootName + ".%i" % fileIndex


def readMMFData(file,HEADER=True,VERBOSE=True):
    """ Reads the data in a MMF file. It returns a list with the MMF header (if HEADER=True) and a numpy array with the values of the data at each grid point.
    Use HEADER=False to cancel returning the header and VERBOSE=False to turn off the messages. """
    
    #read the header and find how many files there are
    header = MMFHeader()
    tempName = file
    if not os.path.isfile(tempName):
        tempName = MMFMultipleFiles( file, 0 )
        if not os.path.isfile(tempName):  throwError( "Cannot find the MMF binary file. There are no '%s' or '%s' files." %(file,tempName) )
    f = open(tempName, 'rb')
    header.fromfile( f )
    f.close()
    if header.noMMFFiles>1:
        for i in range(header.noMMFFiles):
            tempName = MMFMultipleFiles( file, i )
            if not os.path.isfile(tempName):  throwError( "Cannot find the MMF binary file number %i (of %i files) with expected name '%s'." %(i+1,header.noMMFFiles,tempName) )
    
    #read the data from each file
    dataType = header.DataType()
    dataComponents = np.uint64( header.DataComponents() )
    data = np.empty( header.totalGrid * dataComponents, dataType )
    startPosition = 0
    
    
    for i in range(header.noMMFFiles):
        if VERBOSE: print("Reading the data in the MMF file '%s' which is file %i of %i files ... " % (tempName,i+1,header.noMMFFiles))
        tempName = file
        if header.noMMFFiles!=1: tempName = MMFMultipleFiles( file, i )
        if not os.path.isfile(tempName):  throwError( "Cannot find the MMF file number %i with expected name '%s'." %(i+1,tempName) )
        f = open(tempName, 'rb')
        tempHeader = MMFHeader()
        tempHeader.fromfile( f )
        dataSize = np.uint64( tempHeader.totalGrid * dataComponents )
        
        #reading the data
        __buffer1 = np.fromfile( f, bufferType, 1 )[0]
        data = readArrayEntries( data, f, startPosition, dataSize )
        __buffer2 = np.fromfile( f, bufferType, 1 )[0]
        if __buffer1!=__buffer2: throwError( "While reading the MMF data block in file '%s'. The buffer preceding and the buffer after the data do not have the same value (buffer1 = %s, buffer2 = %s while the expected value was %s)." % (tempHeader,__buffer1,__buffer2,dataSize*data[0].nbytes) )
        
    # return the results
    results = []
    if HEADER: results += [header]
    results += [data]
    return results


def writeMMFData(file,header,data,VERBOSE=True,ERROR_CHECK=True):
    """ Writes MMF data to a binary file which has a MMF header and each block of data is preceded and followed by a uint64 integer giving the number of bytes in the data block (for error checking).
    The function takes 3 arguments: name of the output file, the MMF header (class 'MMFHeader') and the data in the form of a numpy array. """
    
    #do some error checking
    if VERBOSE: print("Writing the MMF data to the file '%s' ... " % file, end=' ')
    __temp = header.gridSize[0] * header.gridSize[1] * header.gridSize[2]
    if __temp!=header.totalGrid and ERROR_CHECK:
        throwError( "The total number of grid points in the MMF header is not equal to the product of the grid dimensions along each axis. Total number of grid points is %i while the size along each axis is:" % header.totalGrid, header.gridSize )
    dataComponents = header.DataComponents()
    noDataElements = data.size // dataComponents
    if header.totalGrid!=noDataElements and ERROR_CHECK:
        throwError( "The total number of grid points in the MMF header does not agree with the data length. Number grid points in the header is %i while the data has only %i elements." % (header.totalGrid,noDataElements))
    header.noMMFFiles = np.int32( 1 )
    
    #write the header to file
    f = open(file, 'wb')
    header.tofile(f)
    
    #write the data to file
    data.shape = (-1)
    noBytes = data.size*data[0].nbytes
    __buffer = np.array( [noBytes], dtype=np.uint64 )
    __buffer.tofile( f )
    writeArrayEntries( data, f, 0, data.size )
    __buffer.tofile( f )
    f.close()
    if VERBOSE: print("Done.")
