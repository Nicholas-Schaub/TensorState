# distutils: language=c++
cimport cython
cimport numpy as np
import numpy as np
from libcpp.vector cimport vector

IF UNAME_SYSNAME == "Windows":
    cdef extern from "immintrin.h":
        # Type definitions
        ctypedef int __m128i
        ctypedef int __m256i
        ctypedef float __m256

        # Intrinsic definitions
        __m128i _mm_loadu_si128(__m128i* __A) nogil
        
        __m128i _mm_shuffle_epi8(__m128i __A,__m128i __B) nogil
        
        __m128i _mm_and_si128(__m128i __A,__m128i __B) nogil
        
        __m128i _mm_storeu_si128(__m128i* __A,__m128i __B) nogil

        __m128i _mm_set_epi8(char e15,char e14,char e13,char e12,
                             char e11,char e10,char e9, char e8,
                             char e7, char e6, char e5, char e4,
                             char e3, char e2, char e1, char e0) nogil

        __m128i _mm_set1_epi8(unsigned char __A) nogil
        
        __m128i _mm_set1_epi16(unsigned short __A) nogil
        
        __m128i _mm_cmpgt_epi8(__m128i __A,__m128i __B) nogil
        
        __m128i _mm_setzero_si128() nogil

        __m256 _mm256_loadu_ps(__m256* __A) nogil

        __m256 _mm256_setzero_ps() nogil

        int _mm256_movemask_ps(__m256 __A) nogil

ELSE:
    cdef extern from "x86intrin.h":
        # Type definitions
        ctypedef int __m128i
        ctypedef int __m256i
        ctypedef float __m256

        # Intrinsic definitions
        __m128i _mm_loadu_si128(__m128i* __A) nogil
        
        __m128i _mm_shuffle_epi8(__m128i __A,__m128i __B) nogil
        
        __m128i _mm_and_si128(__m128i __A,__m128i __B) nogil
        
        __m128i _mm_storeu_si128(__m128i* __A,__m128i __B) nogil

        __m128i _mm_set1_epi8(unsigned char __A) nogil

        __m256 _mm256_loadu_ps(__m256* __A) nogil

        __m256 _mm256_setzero_ps() nogil

        int _mm256_movemask_ps(__m256 __A) nogil

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef void __byte_sort(unsigned char [:,:] states,
                      unsigned long long [:] index,
                      unsigned long long start,
                      unsigned long long end,
                      unsigned long long col,
                      long long [:] counts) nogil:

    # Create an index view for fast rearranging of indices
    cdef unsigned long long [:] index_view = index

    # Initialize counting and offset indices
    cdef unsigned long long i
    for i in range(256):
        counts[i] = 0

    cdef unsigned long long offsets[256]

    cdef unsigned long long next_offset[256]

    # Count values
    for i in range(start,end):
        counts[(states[index[i],col])] += 1

    # Calculate cumulative sum and offsets
    cdef unsigned long long num_partitions = 0
    cdef unsigned long long remaining_partitions[256]
    cdef unsigned long long total = 0
    cdef unsigned long long count
    for i in range(256):
        count = counts[i]
        if count:
            offsets[i] = total
            total += count
            remaining_partitions[num_partitions] = i
            num_partitions += 1
        
        next_offset[i] = total

    # Swap index values into place
    cdef unsigned long long val, v
    cdef unsigned long long ind, offset, temp
    for i in range(0,num_partitions-1):
        val = remaining_partitions[i]
        while offsets[val] < next_offset[val]:
            ind = offsets[val]
            v = states[index[start+ind],col]
            if v==val:
                offsets[val] += 1
                continue
            offset = offsets[v]
            offsets[v] += 1
            temp = index_view[start+ind]
            index_view[start+ind] = index_view[start+offset]
            index_view[start+offset] = temp

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef void __lex_sort(unsigned char [:,:] states,
                     unsigned long long [:] index,
                     unsigned long long start,
                     unsigned long long end,
                     unsigned long long col,
                     list bin_edges):

    cdef unsigned long long total = 0
    cdef unsigned long long i
    cdef long long counts[256]

    if col > 0:
        __byte_sort(states, index, start, end, col, counts)
        for i in range(256):
            if counts[i]<=1:
                if counts[i] == 1:
                    bin_edges.append(bin_edges[len(bin_edges)-1] + counts[i])
                total += counts[i]
                continue
            __lex_sort(states, index, start+total, start+total+counts[i], col-1, bin_edges)
            total += counts[i]
    else:
        __byte_sort(states, index, start, end, col, counts)
        for c in counts:
            if c > 0:
                bin_edges.append(bin_edges[len(bin_edges)-1] + c)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cpdef lex_sort(unsigned char [:,:] states,
               unsigned long long state_count):

    index = np.arange(states.shape[0],dtype=np.uint64)

    bin_edges = [0]

    __lex_sort(states,index,0,state_count,states.shape[1]-1,bin_edges)

    return np.asarray(bin_edges,dtype=np.uint64),index

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef void __compress_tensor(const float[:,:] input, unsigned char [:,:] result) nogil:

    # Initialize variables
    cdef __m256 substate
    cdef long long rows, cols, row, col, col_shift, col_floor, i
    cdef unsigned int value_truncate = 0xFFFF
    
    # Get the number of rows and cols in the input
    rows,cols = input.shape[0], input.shape[1]
    
    cdef unsigned char shift = cols % 8
    for col in range(0,cols-shift,8):
        col_shift = col
        col_floor = col_shift//8
        for row in range(rows):
            substate = _mm256_loadu_ps(&input[row,col_shift])
            result[row,col_floor] = _mm256_movemask_ps(substate) ^ value_truncate
    
    if shift > 0:
        col_shift = cols - shift
        col_floor = col_shift//8
        value_truncate = 0
        for i in range(shift):
            value_truncate += 2**i
            
        for row in range(rows):
            substate = _mm256_loadu_ps(&input[row,col_shift])
            mask = _mm256_movemask_ps(substate)
            result[row,col_floor] = (mask ^ 0xFFFF) & value_truncate

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cpdef np.ndarray _compress_tensor(const float [:,:] input):
    # Initialize the output
    rows,cols = input.shape[0], input.shape[1]
    result = np.zeros((rows,int(np.ceil(cols/8))), dtype = np.uint8)

    # Call the nogil method
    __compress_tensor(input,result)

    return result

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef np.ndarray __decompress_tensor(unsigned char [:] input,
                                    long long n_neurons):

    # Initialize the output
    cdef vector[np.uint8_t] output
    output.resize(input.shape[0]//((n_neurons-1)//8 + 1) * n_neurons)

    # Bit shuffle and mask arrays
    cdef __m128i shuffle = _mm_set_epi8(
        0x01, 0x01, 0x01, 0x01,
        0x01, 0x01, 0x01, 0x01,
        0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00
    )
    cdef __m128i mask = _mm_set_epi8(
        0x80, 0x40, 0x20, 0x10,
        0x08, 0x04, 0x02, 0x01,
        0x80, 0x40, 0x20, 0x10,
        0x08, 0x04, 0x02, 0x01,
    )

    cdef long long b, index
    cdef long long num_bytes = (n_neurons)//8
    cdef long long trailing_neurons = n_neurons - num_bytes*8
    cdef long long offset = 0
    if trailing_neurons > 0:
        offset = 1
    cdef __m128i states, shuffle_states, mask_states, nonzeros, ones

    for index in range(0,input.shape[0],num_bytes):
        for b in range(0,num_bytes,2):
        
            if num_bytes - b == 1:
                break

            # Load two bytes of state information (16 neurons)
            states = _mm_set1_epi16(cython.operator.dereference(<unsigned short *> &input[index+b]))

            # Shuffle and apply the mask
            shuffle_states = _mm_shuffle_epi8(states,shuffle)
            mask_states = _mm_and_si128(shuffle_states,mask)

            # Store the result
            _mm_storeu_si128(<__m128i*>&output[(index+b)*8],mask_states)

        if num_bytes % 2 == 1:
            pass
    
    # Turn the uint8 vector into a numpy.ndarray of appropriate size
    result = np.asarray(output,dtype=np.bool_).reshape(-1,n_neurons)

    return result

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cpdef np.ndarray _decompress_tensor(unsigned char [:,:] input, long long n_neurons):

    # Initialize the output
    rows = input.shape[0]

    # Call the nogil method
    result = __decompress_tensor(np.reshape(input,-1),n_neurons)

    return result