# distutils: language=c++
cimport cython
cimport numpy as np
import numpy as np
from cython.parallel import prange

IF UNAME_SYSNAME == "Windows":
    cdef extern from "immintrin.h":
        # Type definitions
        ctypedef int __m128i
        ctypedef int __m256i
        ctypedef float __m256

        # Intrinsic definitions
        unsigned long long _pext_u64(unsigned long long a,unsigned long long b) nogil
        
        __m128i _mm_cmpeq_epi8(__m128i a,__m128i b) nogil
        
        __m128i _mm_shuffle_epi8(__m128i __A,__m128i __B) nogil
        
        __m128i _mm_and_si128(__m128i __A,__m128i __B) nogil
        
        __m128i _mm_storeu_si128(__m128i* __A,__m128i __B) nogil

        __m128i _mm_set_epi8(char e15,char e14,char e13,char e12,
                             char e11,char e10,char e9, char e8,
                             char e7, char e6, char e5, char e4,
                             char e3, char e2, char e1, char e0) nogil
        
        __m128i _mm_set1_epi16(unsigned short __A) nogil

        __m256 _mm256_loadu_ps(__m256* __A) nogil

        unsigned char _mm256_movemask_ps(__m256 __A) nogil

        __m256 _mm256_cmp_ps(__m256 __A, __m256 B, int imm8) nogil

        __m256 _mm256_setzero_ps() nogil

ELSE:
    cdef extern from "x86intrin.h":
        # Type definitions
        ctypedef int __m128i
        ctypedef int __m256i
        ctypedef float __m256

        # Intrinsic definitions
        unsigned long long _pext_u64(unsigned long long a,unsigned long long b) nogil
        
        __m128i _mm_cmpeq_epi8(__m128i a,__m128i b) nogil
        
        __m128i _mm_shuffle_epi8(__m128i __A,__m128i __B) nogil
        
        __m128i _mm_and_si128(__m128i __A,__m128i __B) nogil
        
        __m128i _mm_storeu_si128(__m128i* __A,__m128i __B) nogil

        __m128i _mm_set_epi8(char e15,char e14,char e13,char e12,
                             char e11,char e10,char e9, char e8,
                             char e7, char e6, char e5, char e4,
                             char e3, char e2, char e1, char e0) nogil
        
        __m128i _mm_set1_epi16(unsigned short __A) nogil

        __m256 _mm256_loadu_ps(__m256* __A) nogil

        unsigned char _mm256_movemask_ps(__m256 __A) nogil

        __m256 _mm256_cmp_ps(__m256i __A, __m256i B, int imm8) nogil

        __m256 _mm256_setzero_ps() nogil

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef void __byte_sort(unsigned char [:,:] states,
                      long long [:] index_view,
                      long long start,
                      long long end,
                      long long col,
                      long long *counts) nogil:

    # Initialize counting and offset indices
    cdef long long i
    for i in range(256):
        counts[i] = 0

    cdef long long offsets[256]

    cdef long long next_offset[256]

    # Count values
    for i in range(start,end):
        counts[(states[index_view[i],col])] += 1

    # Calculate cumulative sum and offsets
    cdef long long num_partitions = 0
    cdef long long remaining_partitions[256]
    cdef long long total = 0
    cdef long long count
    for i in range(256):
        count = counts[i]
        if count:
            offsets[i] = total
            total += count
            remaining_partitions[num_partitions] = i
            num_partitions += 1
        
        next_offset[i] = total

    # Swap index values into place
    cdef long long val, v
    cdef long long ind, offset, temp
    for i in range(0,num_partitions-1):
        val = remaining_partitions[i]
        while offsets[val] < next_offset[val]:
            ind = offsets[val]
            v = states[index_view[start+ind],col]
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
cdef long long __lex_sort(unsigned char [:,:] states,
                          long long [:] index,
                          long long start,
                          long long end,
                          long long col,
                          long long [:] bin_edges,
                          long long count) nogil:

    cdef long long total = 0
    cdef long long i, c
    cdef long long counts[256]

    if col > 0:
        __byte_sort(states, index, start, end, col, &counts[0])
        for i in range(256):
            if counts[i]<=1:
                if counts[i] == 1:
                    bin_edges[count] = bin_edges[count-1] + counts[i]
                    count += 1
                total += counts[i]
                continue
            count = __lex_sort(states, index, start+total, start+total+counts[i], col-1, bin_edges, count)
            total += counts[i]
    else:
        __byte_sort(states, index, start, end, col, &counts[0])
        for i in range(256):
            if counts[i] > 0:
                bin_edges[count] = bin_edges[count-1] + counts[i]
                count += 1
    
    return count

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cpdef _lex_sort(unsigned char [:,:] states,
                long long state_count):

    cdef np.ndarray index = np.arange(state_count,dtype=np.int64)
    cdef np.ndarray bin_edges = np.zeros(state_count+1,dtype=np.int64)
    cdef long long count = 1

    count = __lex_sort(states,index,0,state_count,states.shape[1]-1,bin_edges,count)

    return bin_edges[:count],index

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef void __compress_tensor_ps(const float[:,:] input,
                               unsigned char [:,:] result) nogil:

    # Initialize variables
    cdef __m256 substate
    cdef __m256 zeros = _mm256_setzero_ps()
    cdef long long rows, cols, row, col, col_shift, col_floor, i
    cdef unsigned int value_truncate = 0
    
    # Get the number of rows and cols in the input
    rows,cols = input.shape[0], input.shape[1]
    
    cdef unsigned char shift = cols % 8
    for col in range(0,cols-shift,8):
        col_shift = col
        col_floor = col_shift//8
        for row in range(rows):
            substate = _mm256_loadu_ps(&input[row,col_shift])
            substate = _mm256_cmp_ps(substate,zeros,0x0e)
            result[row,col_floor] = _mm256_movemask_ps(substate)
    
    if shift > 0:
        col_shift = cols - shift
        col_floor = col_shift//8
        value_truncate = 0
        for i in range(shift):
            value_truncate += 2**i
            
        for row in range(rows):
            substate = _mm256_loadu_ps(&input[row,col_shift])
            substate = _mm256_cmp_ps(substate,zeros,0x0e)
            mask = _mm256_movemask_ps(substate)
            result[row,col_floor] = mask & value_truncate

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cpdef np.ndarray _compress_tensor_ps(const float [:,:] input):
    # Initialize the output
    cdef long long rows = input.shape[0]
    cdef long long cols = input.shape[1]
    cdef np.ndarray result = np.zeros((rows+1,int(np.ceil(cols/8))), dtype = np.uint8)

    # Call the nogil method
    __compress_tensor_ps(input,result)

    return result[:-1,:]

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
cdef void __compress_tensor_pi8(const unsigned char [:,:] input,
                                unsigned char [:,:] result) nogil:

    # Initialize variables
    cdef long long rows, cols, row, col, col_shift, col_floor, i
    cdef unsigned long long substate, magic
    cdef unsigned char value_truncate = 0xFF

    # Magic number to pack bools
    magic = 0x0101010101010101
    
    # Get the number of rows and cols in the input
    rows,cols = input.shape[0], input.shape[1]
    
    cdef unsigned char shift = cols % 8
    for col in range(0,cols-shift,8):
        col_floor = col//8
        for row in range(rows):
            substate = _pext_u64(cython.operator.dereference(<unsigned long long *> &input[row,col]),magic)
            result[row,col_floor] = substate & 0xFF
    
    if shift > 0:
        col_shift = cols - shift
        col_floor = col_shift//8
        value_truncate = 0
        for i in range(shift):
            value_truncate += 2**i
            
        for row in range(rows):
            substate = _pext_u64(cython.operator.dereference(<unsigned long long *> &input[row,col_shift]),magic)
            result[row,col_floor] = (substate & 0xFF) & value_truncate

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cpdef np.ndarray _compress_tensor_pi8(const unsigned char [:,:] input):
    # Initialize the output
    cdef long long rows = input.shape[0]
    cdef long long cols = input.shape[1]
    cdef np.ndarray result = np.zeros((rows,int(np.ceil(cols/8))), dtype = np.uint8)

    # Call the nogil method
    __compress_tensor_pi8(input,result)

    return result

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
cdef void __decompress_tensor(unsigned char [:] input,
                              long long n_neurons,
                              long long n_states,
                              unsigned char [:] output) nogil:

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

    cdef long long b, index, neuron_index
    cdef long long num_bytes = (n_neurons-1)//8 + 1
    cdef __m128i states, shuffle_states, mask_states, nonzeros

    index = 0 
    while index < n_states:

        neuron_index = index//num_bytes*n_neurons

        for b in range(0,num_bytes,2):

            # Load two bytes of state information (16 neurons)
            states = _mm_set1_epi16(cython.operator.dereference(<unsigned short *> &input[index+b]))

            # Shuffle and apply the mask
            shuffle_states = _mm_shuffle_epi8(states,shuffle)
            mask_states = _mm_and_si128(shuffle_states,mask)
            nonzeros = _mm_cmpeq_epi8(mask_states,mask)

            # Store the result
            _mm_storeu_si128(<__m128i*>&output[neuron_index+b*8],nonzeros)

        index += num_bytes

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cpdef np.ndarray _decompress_tensor(unsigned char [:,:] input, long long n_neurons):

    # Initialize the output
    cdef long long n_states = input.shape[0]*input.shape[1]
    cdef np.ndarray result = np.zeros(input.shape[0]*n_neurons+16,dtype=np.bool_)
    cdef unsigned char [:] output = result

    # Call the nogil method
    __decompress_tensor(np.reshape(input,-1),n_neurons,n_states,output)

    return result[:input.shape[0]*n_neurons].reshape(input.shape[0],n_neurons)