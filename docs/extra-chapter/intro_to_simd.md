---
layout: default
title: (WIP) Intro to practical SIMD for graphics
parent: Extra Chapter
nav_order: 45
---

# SIMD Overview
As CPUs hit the ghz barrier, development tried to move them to higher degrees of parallelism. One axis of parallelism is building multiple cores into the same CPU, which is explained on here [Multithreading for game engines]({{ site.baseurl }}{% link docs/extra-chapter/multithreading.md %}) Multithreading is great if you have various different things and want to speed them up, but what if you want to speed up a single operation or something small like a matrix multiplication? CPU vendors began implementing SIMD instructions (Single Instruction, Multiple Data) which, instead of adding one number to another, add a pack of 4 numbers to another pack of 4 numbers. Now, adding numbers is 4x faster, as long as you can find a way to pack those numbers, of course. 

SIMD programming, or vector programming, is the same as normal programming, but instead of dealing with values one by one, you deal with them in groups, using much larger CPU registers. The size of those registers can vary, and it limits how many numbers (or data) you can pack for a single instruction to process.

Over the years, many SIMD instruction sets have been released as CPUs evolved. Each of them is different, which is one of the biggest problems on SIMD programming, as you can easily write code that works on one CPU, but doesnt in other because the instruction set supported is different. On x86, used in the big consoles (except switch) and PCs, we have SSE, AVX, and AVX512. On ARM (phones, nintendo switch) we have NEON and SVE, and on RiscV cpus we see RVV. Each of those have their own gotchas and properties, so generally an algorithm will have to be written for the sets that are supported on whatever cpus you are targetting. 

## X86 (PC) SIMD sets
- SSE4 : According to Steam Hardware survey, supported on 99.78% of PCs. Good as a baseline. This set is 128 bits per register, which means its 4-wide on floating point operations. 
- AVX : Essentially a wider version of SSE, it moves into 256 bits per register, for 8-wide floating point operations. AVX1 is supported by 97.24% of gaming PCs, and AVX2 is supported by 95.03% of PCs. AVX2 mostly adds a few instructions for shuffling data around, and 8-wide integer operations. Because AVX was designed to run on SSE4 compatible CPU math units, a lot of the operations have a weird "half and half" execution, where it applies separately to the first 4 numbers and the next 4 numbers. This will be something to take into mind when programming it. Some AVX cpus have support for the optional "FMA" extension, which adds the possibility of multiplying and adding values as 1 operation, which can 2x the speed of many common graphics operations. 
- AVX512 : While it keeps the AVX name, its a complete rewrite from the AVX1-2 instruction set, with fully different instructions. This is currently the most advanced shipped instruction set, with 512-wide registers that fit 16 floats at a time, but also a extensive set of masking systems and handy instructions that make writing advanced algorithms much, much easier than in AVX2. Sadly, its at sub 20% support on gaming PCs, thanks to Intel dropping it from many consumer cpus. Not relevant for games due to low support.

## ARM SIMD sets
- NEON : Seen on pretty much every single phone cpu and the nintendo switch 1 and 2, this is a 128 bit instruction set, for 4-wide float operations. Its pretty much a direct improvement over SSE4, but doing similar things. Must have for game-devs that target mobile hardware.
- SVE : Scalable vectors. Bleeding edge, we are barely seeing CPUs with it. This does not use fixed register size, but instead you query the CPU feature flags for what register size it uses, and then it can scale from 4-wide math units like NEON into 16-wide math units like AVX512. Irrelevant to game devs, no consumer hardware supports it

## RISCV SIMD sets
- RVV : Equivalent to SVE but for RiscV cpus. Only used in some prototype hardware with basically no consumer hardware using this. 

## What instruction set to target?
For a high end modern game, targetting AVX2 as the default feature level is fine. You will lose 5% of steam consumers that are on low end Intel CPUs, but its going to be able to target PS5, Xbox, and every decent gaming PC built in the last 15 years. If you are a indie game, you want to drop to AVX1 as many indie players will expect their very low end computer to play the game. SSE4 is virtually guaranteed to run on every PC.
If you target mobile or nintendo switch, then you use NEON instruction set. All other instruction set versions are irrelevant to games either due to small consumer base (AVX512), or it being non-consumer to begin with (SVE, RVV)

# How to program SIMD
So, with the history lesson over, how do we actually make use of this. The answer is to use either intrinsics, or a library that abstracts over them.

 Intrinsics expose the instructions seen on the simd set, and you can use them to program your algorithms. They are a considerable pain to use, because they are very low level and directly naming things from the assembly manual. Intrinsics are also "fixed" to a given feature level, so you will need to write your algorithm multiple times to target different feature sets such as NEON for Switch vs AVX2 for PC. Libraries like [XSIMD](https://github.com/xtensor-stack/xsimd) abstract over intrinsics, and you can use them to write the code once and target multiple platforms.The downside of using libraries like this is that as they are abstracting over multiple feature sets, there are a lot of missing operations as they will work on a common denominator of features. 

Another option is to use [ISPC](https://ispc.github.io/ispc.html). This is a compiler that takes a subset of C with a few extra keywords, and compiles it into highly optimized SIMD code. This lets you essentially write compute shader type logic but for CPU. It will not be as fast as intrinsics because the compiler is not all-knowing, but it will give very good speedups over normal C/Cpp code. Its also much easier and faster to write than going directly to intrinsics, so it may be worth it just from the better maintenance and using it in more places of the codebase.

You can also have your normal compiler output vector operations, if you give the compiler the minimum SIMD target your program can use. You want to use this with either AVX1 or SSE4, and then the compiler will autovectorize some stuff. Because vector operations are complicated, you will quickly find that compilers trying to autovectorize C code is incredibly unreliable, and thus not a way to optimize anything. This is more of a nice bonus to get a bit of speedup across the whole codebase.

Direct assembly is an option too for some *very* heavy kernels. The FFMpeg video encoding library does this, but its only really something to keep for the hottest of hot code. You can often get a 20% speedup with handwritten simd assembly instead of using intrinsics, as you can directly control the registers and timing of instructions. This is the most advanced option, and generally never used in game dev due to the maintenance burden and how hard it is to actually beat the compiler.
We have a hierarchy here of hardest but fastest, to simplest but slowest. Assembly -> Intrinsics -> SIMD Libraries -> ISPC -> enabling vectors in the compiler
This article will focus on SIMD intrinsics, as its a great way to learn what these operations are doing without the abstraction of the libraries.

# Hello SIMD
So whats the simplest thing to use SIMD for? Every tutorial always starts with the same thing, adding 2 arrays of numbers together, or multiplying them together, so lets just do that.

```cpp

//Adds B to A. The arrays must have the same size
void add_arrays(float* A, float* B, size_t count){
    for(size_t i = 0; i < count; i++){
        A[i] += B[i]; 
    }
}
```

This is a very simple function where we have 2 arrays of floats and we add the second one to the first. We will be using SSE4 for this, which does operations 4 numbers at a time, so to think of it, lets unroll this code in sets of 4
```cpp

//Adds B to A. The arrays must have the same size
void add_arrays_unroll4(float* A, float* B, size_t count){
    size_t i = 0;
    for(i; i < count; i+= 4){
        A[i+0] += B[i+0]; 
        A[i+1] += B[i+1]; 
        A[i+2] += B[i+2]; 
        A[i+3] += B[i+3]; 
    }

    //loop terminator. What if the loop isnt a multiple of 4?
    for(i; i < count; i++){
        A[i] += B[i]; 
    }
}

```

We immediately run into a problem here. If we are doing operations in groups of 4, how do we deal with a non-4 divisible workload? We have to add a scalar (non-vector) path at the end to do the stragglers. This happens the same with any simd code, and a common way of dealing with it is padding the workload to a multiple or 4 or 8. 

With the function unrolled, lets try to look into the [Intel SIMD reference](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#techs=SSE_ALL) document, to try to find what we could be using here.  As our goal is to target SSE for this, we tick the "SSE Family" checkbox at the left. Now we can search for `add` and see there are a lot of possible operations. 

I heavily recomend you look at the instructions i mention here in the documentation, as this documentation is the SIMD bible and will tell you what cpus support each instruction and how fast the instruction is.

The one we actually want is `_mm_add_ps`. This takes 2 `__m128` values (referencing the 128 bit registers) and adds them together. 
This isnt taking a float* or similar, so we need to first load our values into SIMD registers. There are many ways of doing it, but the one we want here is `_mm_load_ps`which takes a float* and loads it into a vector. In the same way, we want the store too as `_mm_store_ps`, to take our vector register with the math operation, and save it on the memory.

Using them, we have our vector addition, and it looks like this.

```cpp

//need to grab intrinsics header
#include <immintrin.h>

//Adds B to A. The arrays must have the same size
void add_arrays_sse(float* A, float* B, size_t count){
    size_t i = 0;
    for(i; i < count; i+= 4){
        __m128 vA = _mm_load_ps(A + i);
        __m128 vB = _mm_load_ps(B + i);

        _mm_store_ps(_mm_add_ps(vA,vB), A + i);
    }

    //loop terminator. What if the loop isnt a multiple of 4?
    for(i; i < count; i++){
        A[i] += B[i]; 
    }
}
```

We are using a bit of vector math to index the arrays for the `_mm_load_ps`, loading the data into the simd vectors, and then doing our math and storing it. This loop should be almost 4x faster than the scalar version.

But this is the SSE version, which is 4-wide. What if we want to do it as 8-wide using AVX? We go back to the intrinsic reference, and find the avx version of the same load, store, and add functions. This gives us `_mm256_load_ps`, `_mm256_store_ps`, and ` _mm256_add_ps`, same thing, but using `__mm256` vector variables instead of the `__mm128` ones, so 8 floats, not 4. Loop looks like this

```cpp

//need to grab intrinsics header
#include <immintrin.h>

//Adds B to A. The arrays must have the same size
void add_arrays_avx(float* A, float* B, size_t count){
    size_t i = 0;
    for(i; i < count; i+= 8){
        __m256 vA = _mm256_load_ps(A + i);
        __m256 vB = _mm256_load_ps(B + i);

        _mm256_store_ps(_mm256_add_ps(vA,vB), A + i);
    }

    //loop terminator. What if the loop isnt a multiple of 8?
    for(i; i < count; i++){
        A[i] += B[i]; 
    }
}
```

In general, for most basic operations, AVX is just SSE but doubled. There are some advanced operations that are handy that use the smaller 128 registers, and we can code as 8-wide for a speedup if it fits the algorithm.

If you want to see how NEON would work, you look at the ARM reference instead of the Intel one [NEON Intrinsics](https://developer.arm.com/documentation/den0018/a/NEON-Intrinsics). You will find the actual instruction table on the left menu, `Neon Intrinsics Reference` section  The NEON version of this addition loop looks like this

```cpp
#include <arm_neon.h>

//Adds B to A. The arrays must have the same size
void add_arrays_neon(float* A, float* B, size_t count){
    size_t i = 0;
    for(i; i < count; i+= 4){
        float32x4_t  vA = vld1_f32(A + i);
        float32x4_t  vB = vld1_f32(B + i);

        vst1_f32(vadd_f32(vA,vB), A + i);
    }

    //loop terminator. What if the loop isnt a multiple of 4?
    for(i; i < count; i++){
        A[i] += B[i]; 
    }
}
``` 
Its the same as the SSE version, but with the variable types and function names all changed into a completely different style.

# 4x4 Matmul
There is limited usefulness to just adding 2 large arrays of floats together. Lets now look at something that is an actual real use case for vectors, Multiplying 4x4 matrices with each other. This is something that is seen constantly on transform systems and its a hot path of animation code. Its also a good use case of using SIMD intrinsics to make a single thing faster, instead of running 4 things at the same time. 

When writing SIMD, we generally have 2 main options on how to deal with the operations. We can do what is known as `vertical` SIMD, where we use the SIMD operations to optimize 1 thing, or we can do `horizontal` SIMD where we instead use SIMD to do the same operation but N times in parallel. The addition loop above is a example of horizontal simd, as we are doing the same operation (add 2 numbers) but 4 or 8 times at once. For this matrix multiplication, we could do it in horizontal too, but that means we would need to multiply 8 matrices by another 8 matrices, or 8 matrices by a single one. This has limited usage compared to just multiplying 1 matrix faster, so lets look at what a matrix multiply does, and how can we make it faster.

To mirror the vulkan tutorial, we will be using the GLM library matrices. GLM already has sse4 optimized matrixmul somewhere in the codebase, but we are going to create our own version of it. Lets begin by writing the non-simd loop and see what kind of logic we are dealing with.

```cpp
void matmul4x4(const mat4x4& m1, const mat4x4& m2, mat4x4& out){

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            for (int k = 0; k < 4; k++) {
                result[i][j] += m1[k][j] * m2[i][k];
            }
        }
    }
}
```

We have a set of nested for loops, so its not quite so clear how can we vectorize this. To vectorize it, we need to find a way of writing the logic so that we can write the data as direct rows of 4 float at a time (the `j` axis on the result). SIMD loads and stores are contiguous, so for our algorithms we always need to find a way to arrange the calculations to load contiguous values and store contiguous values. Its also important to remember that moving data into vector registers can be a bit slow, so we must find a way to load as much data into simd variables as possible, and do all the operations on them directly, then store the result out. If we are moving from scalar land into vector land constnatly we end up losing all the performance wins from vectors.

We need to go at a bit higher level of abstraction to think about the algorithm properly, and think what operations are needed here for vectorization.
In a matrix multiplication, for each of the elements in the matrix, we are doing a dot product of the row of one matrix and the column of another. One possible way of vectorizing a matrix mul could take advantage of it, loading one column from a matrix, a row from the other one, and calling `_mm_dp_ps` to do a dot product. This is not that recomended for a case like a matrixmul because SIMD operations that work across the lanes of the simd register end to be high in latency and slower than simd operations that are per-lane fully parallel. 

But we could try to do 4 dot products at once. Looking at the algorithm, we can grab a column from one matrix, then grab the 4 rows of the other one, and multiply everything together like that, doing 4 dot products at a time and then writing 4 values at a time to the result matrix.

To load an entire matrix, we can do 4 load calls, one per row, and just store them as 4 variables. Or rows are stored contiguously, so its a good fit to do it this way.

```cpp
    //load the entire m1 matrix to simd registers
    __m128  vx = _mm_load_ps(&m1[0][0]);
    __m128  vy = _mm_load_ps(&m1[1][0]);
    __m128  vz = _mm_load_ps(&m1[2][0]);
    __m128  vw = _mm_load_ps(&m1[3][0]);
```

The other piece we need is to load a single column into 4 simd registers, with the numbers duplicated. We can do multiple scalar loads, but thats actually a bit slower than loading once and then rearranging the memory a bit with shuffles.

```cpp
__m128 col = _mm_load_ps(&m2[i][0]);

__m128 x = _mm_shuffle_ps(col,col, _MM_SHUFFLE(0, 0, 0, 0));
__m128 y = _mm_shuffle_ps(col,col, _MM_SHUFFLE(1, 1, 1, 1));
__m128 z = _mm_shuffle_ps(col,col, _MM_SHUFFLE(2, 2, 2, 2));
__m128 w = _mm_shuffle_ps(col,col, _MM_SHUFFLE(3, 3, 3, 3));
```

The shuffle instruction lets us move around the values of a simd register. This snippet loads the column once, and then converts it into 4 simd registers, with the first being "x,x,x,x", second being "y,y,y,y" and so on. This is a very common technique, and if you target avx you can do it with the `_mm_permute_ps` which works similar.

The last part of the algorithm is to perform the actual dot product calculation, 4 of them at once, and store the result

```cpp
 __m128 result_col = _mm_mul_ps(vx, x);
        result_col = _mm_add_ps(result_col, _mm_mul_ps(vy, y));
        result_col = _mm_add_ps(result_col, _mm_mul_ps(vz, z));
        result_col = _mm_add_ps(result_col, _mm_mul_ps(vw, w));

        _mm_store_ps(&out[i][0], result_col);
```

The full algorithm looks like this.
```cpp
void matmul4x4(const mat4x4& m1, const mat4x4& m2, mat4x4& out){
    //load the entire m1 matrix to simd registers
    __m128  vx = _mm_load_ps(&m1[0][0]);
    __m128  vy = _mm_load_ps(&m1[1][0]);
    __m128  vz = _mm_load_ps(&m1[2][0]);
    __m128  vw = _mm_load_ps(&m1[3][0]);

     for (int i = 0; i < 4; i++) {
        //load a row and widen it
        __m128 col = _mm_load_ps(&m2[i][0]);

        __m128 x = _mm_shuffle_ps(col,col, _MM_SHUFFLE(0, 0, 0, 0));
        __m128 y = _mm_shuffle_ps(col,col, _MM_SHUFFLE(1, 1, 1, 1));
        __m128 z = _mm_shuffle_ps(col,col, _MM_SHUFFLE(2, 2, 2, 2));
        __m128 w = _mm_shuffle_ps(col,col, _MM_SHUFFLE(3, 3, 3, 3));

        //dot products
        __m128 result_col = _mm_mul_ps(vx, x);
        result_col = _mm_add_ps(result_col, _mm_mul_ps(vy, y));
        result_col = _mm_add_ps(result_col, _mm_mul_ps(vz, z));
        result_col = _mm_add_ps(result_col, _mm_mul_ps(vw, w));

        _mm_store_ps(&out[i][0], result_col);
    }
}
```

There are actually a fair few ways of doing a 4x4 matmul, as it can change depending on how your matrix is laid out in memory. This version works for how GLM does things, and its similar to how they themselves vectorize it.

