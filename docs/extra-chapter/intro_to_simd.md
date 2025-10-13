---
layout: default
title: Intro to practical SIMD for graphics
parent: Extra Chapter
nav_order: 45
---

# SIMD Overview
As CPUs hit the GHz barrier, development tried to move them to higher degrees of parallelism. One axis of parallelism is building multiple cores into the same CPU, which is explained here: [Multithreading for game engines]({{ site.baseurl }}{% link docs/extra-chapter/multithreading.md %}). Multithreading is great if you have various different things and want to speed them up, but what if you want to speed up a single operation or something small like a matrix multiplication? CPU vendors began implementing SIMD instructions (Single Instruction, Multiple Data) which, instead of adding one number to another, add a pack of 4 numbers to another pack of 4 numbers. Now, adding numbers is 4x faster, as long as you can find a way to pack those numbers, of course. 

SIMD programming, or vector programming, is the same as normal programming, but instead of dealing with values one by one, you deal with them in groups, using much larger CPU registers. The size of those registers can vary, and it limits how many numbers (or data) you can pack for a single instruction to process.

Over the years, many SIMD instruction sets have been released as CPUs evolved. Each of them is different, which is one of the biggest problems in SIMD programming, as you can easily write code that works on one CPU, but doesn't on another because the instruction set supported is different. On x86, used in the big consoles (except Switch) and PCs, we have SSE, AVX, and AVX512. On ARM (phones, Nintendo Switch) we have NEON and SVE, and on RISC-V CPUs we see RVV. Each of those have their own gotchas and properties, so generally an algorithm will have to be written for the sets that are supported on whatever CPUs you are targeting. 

## X86 (PC) SIMD sets
- SSE4 : According to Steam Hardware survey, supported on 99.78% of PCs. Good as a baseline. This set is 128 bits per register, which means it's 4-wide for floating-point operations. 
- AVX : Essentially a wider version of SSE, it moves into 256 bits per register, for 8-wide floating point operations. AVX1 is supported by 97.24% of gaming PCs, and AVX2 is supported by 95.03% of PCs. AVX2 mostly adds a few instructions for shuffling data around, and 8-wide integer operations. Because AVX was designed to run on SSE4-compatible CPU math units, a lot of the operations have a weird "half and half" execution, where it applies separately to the first 4 numbers and the next 4 numbers. This will be something to take into mind when programming it. Some AVX CPUs have support for the optional "FMA" extension, which adds the possibility of multiplying and adding values as 1 operation, which can 2x the speed of many common graphics operations. 
- AVX512 : While it keeps the AVX name, it's a complete rewrite from the AVX1-2 instruction set, with fully different instructions. This is currently the most advanced shipped instruction set, with 512-bit registers that fit 16 floats at a time, but also an extensive set of masking systems and handy instructions that make writing advanced algorithms much, much easier than in AVX2. Sadly, it's at sub-20% support on gaming PCs, thanks to Intel dropping it from many consumer CPUs. Not relevant for games due to low support.

## ARM SIMD sets
- NEON : Seen on pretty much every single phone CPU and the Nintendo Switch 1 and 2, this is a 128-bit instruction set, for 4-wide float operations. It's pretty much a direct improvement over SSE4, but doing similar things. Must-have for game devs that target mobile hardware.
- SVE : Scalable vectors. Bleeding edge, we are barely seeing CPUs with it. This does not use fixed register size, but instead you query the CPU feature flags for what register size it uses, and then it can scale from 4-wide math units like NEON into 16-wide math units like AVX512. Irrelevant to game devs, no consumer hardware supports it.

## RISC-V SIMD sets
- RVV : Equivalent to SVE but for RISC-V CPUs. Only used in some prototype hardware with basically no consumer hardware using this. 

## What instruction set to target?
For a high-end modern game, targeting AVX2 as the default feature level is fine. You will lose 5% of Steam consumers that are on low-end Intel CPUs, but it's going to be able to target PS5, Xbox, and every decent gaming PC built in the last 15 years. If you are an indie game, you want to drop to AVX1 as many indie players will expect their very low-end computer to play the game. SSE4 is virtually guaranteed to run on every PC.
If you target mobile or Nintendo Switch, then you use NEON instruction set. All other instruction set versions are irrelevant to games either due to small consumer base (AVX512), or it being non-consumer to begin with (SVE, RVV).

# How to program SIMD
So, with the history lesson over, how do we actually make use of this. The answer is to use either intrinsics, or a library that abstracts over them.

 Intrinsics expose the instructions seen on the SIMD set, and you can use them to program your algorithms. They are a considerable pain to use, because they are very low level and directly naming things from the assembly manual. Intrinsics are also "fixed" to a given feature level, so you will need to write your algorithm multiple times to target different feature sets such as NEON for Switch vs AVX2 for PC. Libraries like [XSIMD](https://github.com/xtensor-stack/xsimd) abstract over intrinsics, and you can use them to write the code once and target multiple platforms. The downside of using libraries like this is that as they are abstracting over multiple feature sets, there are a lot of missing operations as they will work on a common denominator of features. 

Another option is to use [ISPC](https://ispc.github.io/ispc.html). This is a compiler that takes a subset of C with a few extra keywords, and compiles it into highly optimized SIMD code. This lets you essentially write compute-shader-type logic but for CPU. It will not be as fast as intrinsics because the compiler is not all-knowing, but it will give very good speedups over normal C/C++ code. It's also much easier and faster to write than going directly to intrinsics, so it may be worth it just from the better maintenance and using it in more places of the codebase.

You can also have your normal compiler output vector operations, if you give the compiler the minimum SIMD target your program can use. You want to use this with either AVX1 or SSE4, and then the compiler will autovectorize some stuff. Because vector operations are complicated, you will quickly find that compilers trying to autovectorize C code is incredibly unreliable, and thus not a way to optimize anything. This is more of a nice bonus to get a bit of speedup across the whole codebase.

Direct assembly is an option too for some *very* heavy kernels. The FFmpeg video encoding library does this, but it's only really something to keep for the hottest of hot code. You can often get a 20% speedup with handwritten SIMD assembly instead of using intrinsics, as you can directly control the registers and timing of instructions. This is the most advanced option, and generally never used in game dev due to the maintenance burden and how hard it is to actually beat the compiler.
We have a hierarchy here from hardest but fastest, to simplest but slowest: Assembly -> Intrinsics -> SIMD Libraries -> ISPC -> enabling vectors in the compiler
This article will focus on SIMD intrinsics, as it's a great way to learn what these operations are doing without the abstraction of the libraries.

# Hello SIMD
So what's the simplest thing to use SIMD for? Every tutorial always starts with the same thing, adding 2 arrays of numbers together, or multiplying them together, so let's just do that.

```cpp

// Adds B to A. The arrays must have the same size
void add_arrays(float* A, float* B, size_t count){
    for(size_t i = 0; i < count; i++){
        A[i] += B[i];
    }
}
```

![map]({{site.baseurl}}/diagrams/simd/scalar_add.svg)

This is a very simple function where we have 2 arrays of floats and we add the second one to the first. We will be using SSE4 for this, which does operations 4 numbers at a time, so to think of it, let's unroll this code in sets of 4
```cpp

//Adds B to A. The arrays must have the same size
void add_arrays_unroll4(float* A, float* B, size_t count){
    size_t i = 0;
    for(; i < count; i+= 4){
        A[i+0] += B[i+0]; 
        A[i+1] += B[i+1]; 
        A[i+2] += B[i+2]; 
        A[i+3] += B[i+3]; 
    }

    //loop terminator. What if the loop isnt a multiple of 4?
    for(; i < count; i++){
        A[i] += B[i]; 
    }
}

```

We immediately run into a problem here. If we are doing operations in groups of 4, how do we deal with a non-4-divisible workload? We have to add a scalar (non-vector) path at the end to do the stragglers. This happens the same with any SIMD code, and a common way of dealing with it is padding the workload to a multiple of 4 or 8.

With the function unrolled, let's try to look into the [Intel SIMD reference](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#techs=SSE_ALL) document, to try to find what we could be using here. As our goal is to target SSE for this, we tick the "SSE Family" checkbox at the left. Now we can search for `add` and see there are a lot of possible operations.

I heavily recommend you look at the instructions I mention here in the documentation, as this documentation is the SIMD bible and will tell you what CPUs support each instruction and how fast the instruction is.

The one we actually want is `_mm_add_ps`. This takes 2 `__m128` values (referencing the 128-bit registers) and adds them together.
This isn't taking a float* or similar, so we need to first load our values into SIMD registers. There are many ways of doing it, but the one we want here is `_mm_loadu_ps` which takes a float* and loads it into a vector. In the same way, we want the store too as `_mm_storeu_ps`, to take our vector register with the math operation, and save it back to memory.

There are 2 versions of load and store. For loads: `_mm_load_ps` (aligned) and `_mm_loadu_ps` (unaligned). For stores: `_mm_store_ps` (aligned) and `_mm_storeu_ps` (unaligned). The aligned versions require the data to be aligned to 16-byte boundaries. Especially in older CPUs, the aligned versions will be faster, so if you know that the data will be aligned to 128 bits, you want to use the aligned versions. On most modern CPUs there isn't a difference, so unless you know your data is aligned it's better to use the unaligned versions. As this function takes an arbitrary float array, we don't have the guarantee for alignment.

Using them, we have our vector addition, and it looks like this.

```cpp

//need to grab intrinsics header
#include <immintrin.h>

//Adds B to A. The arrays must have the same size
void add_arrays_sse(float* A, float* B, size_t count){
    size_t i = 0;
    for(; i < count; i+= 4){
        __m128 vA = _mm_loadu_ps(A + i);
        __m128 vB = _mm_loadu_ps(B + i);

        _mm_storeu_ps(A + i, _mm_add_ps(vA,vB));
    }

    //loop terminator. What if the loop isn't a multiple of 4?
    for(; i < count; i++){
        A[i] += B[i]; 
    }
}
```


![map]({{site.baseurl}}/diagrams/simd/wide_add.svg)

We are using a bit of vector math to index the arrays for the `_mm_loadu_ps`, loading the data into the SIMD vectors, and then doing our math and storing it. This loop should be almost 4x faster than the scalar version.

But this is the SSE version, which is 4-wide. What if we want to do it as 8-wide using AVX? We go back to the intrinsic reference, and find the AVX version of the same load, store, and add functions. This gives us `_mm256_load_ps`, `_mm256_store_ps`, and ` _mm256_add_ps`, same thing, but using `__mm256` vector variables instead of the `__mm128` ones, so 8 floats, not 4. Loop looks like this

```cpp

//need to grab intrinsics header
#include <immintrin.h>

//Adds B to A. The arrays must have the same size
void add_arrays_avx(float* A, float* B, size_t count){
    size_t i = 0;
    for(; i < count; i+= 8){
        __m256 vA = _mm256_loadu_ps(A + i);
        __m256 vB = _mm256_loadu_ps(B + i);

        _mm256_storeu_ps(A + i, _mm256_add_ps(vA,vB));
    }

    //loop terminator. What if the loop isn't a multiple of 8?
    for(; i < count; i++){
        A[i] += B[i]; 
    }
}
```

In general, for most basic operations, AVX is just SSE but doubled. There are some advanced operations that are handy that use the smaller 128 registers, and we can code as 8-wide for a speedup if it fits the algorithm.

If you want to see how NEON would work, you look at the ARM reference instead of the Intel one: [NEON Intrinsics](https://developer.arm.com/documentation/den0018/a/NEON-Intrinsics). You will find the actual instruction table on the left menu, `Neon Intrinsics Reference` section. The NEON version of this addition loop looks like this

```cpp
#include <arm_neon.h>

//Adds B to A. The arrays must have the same size
void add_arrays_neon(float* A, float* B, size_t count){
    size_t i = 0;
    for(; i < count; i+= 4){
        float32x4_t  vA = vld1q_f32(A + i);
        float32x4_t  vB = vld1q_f32(B + i);

        vst1q_f32(A + i, vaddq_f32(vA,vB));
    }

    //loop terminator. What if the loop isn't a multiple of 4?
    for(; i < count; i++){
        A[i] += B[i]; 
    }
}
``` 
It's the same as the SSE version, but with the variable types and function names all changed into a completely different style.

# 4x4 Matmul
There is limited usefulness to just adding 2 large arrays of floats together. Let's now look at something that is an actual real use case for vectors: multiplying 4x4 matrices with each other. This is something that is seen constantly in transform systems and it's a hot path of animation code. It's also a good use case of using SIMD intrinsics to make a single thing faster, instead of running 4 things at the same time. 

When writing SIMD, we generally have 2 main options on how to deal with the operations. We can do what is known as `vertical` SIMD, where we use the SIMD operations to optimize 1 thing, or we can do `horizontal` SIMD where we instead use SIMD to do the same operation but N times in parallel. The addition loop above is an example of horizontal SIMD, as we are doing the same operation (add 2 numbers) but 4 or 8 times at once. For this matrix multiplication, we could do it in horizontal too, but that means we would need to multiply 8 matrices by another 8 matrices, or 8 matrices by a single one. This has limited usage compared to just multiplying 1 matrix faster, so let's look at what a matrix multiply does, and how can we make it faster.

To mirror the Vulkan tutorial, we will be using the GLM library matrices. GLM already has SSE4-optimized matrixmul somewhere in the codebase, but we are going to create our own version of it. Let's begin by writing the non-SIMD loop and see what kind of logic we are dealing with.

```cpp
void matmul4x4(const mat4x4& m1, const mat4x4& m2, mat4x4& out){

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            out[i][j] = 0; // clear the matrix
            for (int k = 0; k < 4; k++) {
                out[i][j] += m1[k][j] * m2[i][k];
            }
        }
    }
}
```

We have a set of nested for loops, so it's not quite so clear how can we vectorize this. To vectorize it, we need to find a way of writing the logic so that we can write the data as direct rows of 4 floats at a time (the `j` axis on the result). SIMD loads and stores are contiguous, so for our algorithms we always need to find a way to arrange the calculations to load contiguous values and store contiguous values. It's also important to remember that moving data into vector registers can be a bit slow, so we must find a way to load as much data into SIMD variables as possible, and do all the operations on them directly, then store the result out. If we are moving from scalar land into vector land constantly we end up losing all the performance wins from vectors.

This diagram shows what is going on in here. Visualizing and drawing the algorithm helps a lot when doing vectorization, to identify patterns on the data.
![map]({{site.baseurl}}/diagrams/simd/vkguide_simd_mat4_base.svg)

In a matrix multiplication, for each of the elements in the matrix, we are doing a dot product of the row of one matrix and the column of another. In this algorithm, we have a couple ways of parallelizing it.

We could try to parallelize the dot product itself for each individual result element. This means we need to do 16 loops, and on each, load a column vector and a row vector, then do `_mm_dp_ps` to do the dot product itself. Due to the data patterns, this is not a good option. Too many instructions total and too much data shuffling due to the column vectors. If the second matrix was transposed, this could be a better option.

The other option would be to parallelize the dot product calculation, and do 4 dot products at once, for filling an entire row of the result.

Looking at the matrix multiplication, we can see that for one row of the result, it will multiply the column of that value with the shared row of the other matrix. Visualizing it, it looks like this, which is what we want to do to calculate this matrix multiplication.

![map]({{site.baseurl}}/diagrams/simd/vkguide_simd_mat4_simd.svg)


Let's implement that. First, we will load the entire first matrix to variables, as it's going to be shared for all the calculations.

```cpp
    //load the entire m1 matrix to SIMD registers
    __m128  a = _mm_load_ps(&m1[0][0]);
    __m128  b = _mm_load_ps(&m1[1][0]);
    __m128  c = _mm_load_ps(&m1[2][0]);
    __m128  d = _mm_load_ps(&m1[3][0]);
```

SSE does not have scalar by vector multiplication, so if we want to multiply a vector by a single number, we will need to convert that single number into a vector, with the value duplicated. To do that, we are going to load one row from m2 in a single SIMD load, and then use shuffling to create 4 vectors with the x,y,z,w values each duplicated across all lanes.

```cpp
__m128 col = _mm_load_ps(&m2[i][0]);

__m128 x = _mm_shuffle_ps(col,col, _MM_SHUFFLE(0, 0, 0, 0));
__m128 y = _mm_shuffle_ps(col,col, _MM_SHUFFLE(1, 1, 1, 1));
__m128 z = _mm_shuffle_ps(col,col, _MM_SHUFFLE(2, 2, 2, 2));
__m128 w = _mm_shuffle_ps(col,col, _MM_SHUFFLE(3, 3, 3, 3));
```

The shuffle instruction lets us move around the values of a SIMD register. This snippet loads the row once, and then converts it into 4 SIMD registers, with the first being "x,x,x,x", second being "y,y,y,y" and so on. This is a very common technique, and if you target AVX you can do it with the `_mm_permute_ps` which works similarly. In AVX, it's also possible to directly load 1 float into a wide register with the `_mm_broadcast_ss` intrinsic. 

The last part of the algorithm is to perform the actual dot product calculation, 4 of them at once, and store the result:

```cpp
 __m128 result_col = _mm_mul_ps(a, x);
        result_col = _mm_add_ps(result_col, _mm_mul_ps(b, y));
        result_col = _mm_add_ps(result_col, _mm_mul_ps(c, z));
        result_col = _mm_add_ps(result_col, _mm_mul_ps(d, w));

        _mm_store_ps(&out[i][0], result_col);
```

The full algorithm looks like this.
```cpp
void matmul4x4(const mat4x4& m1, const mat4x4& m2, mat4x4& out){
    //load the entire m1 matrix to SIMD registers
    __m128  a = _mm_load_ps(&m1[0][0]);
    __m128  b = _mm_load_ps(&m1[1][0]);
    __m128  c = _mm_load_ps(&m1[2][0]);
    __m128  d = _mm_load_ps(&m1[3][0]);

     for (int i = 0; i < 4; i++) {
        //load a row and widen it
        __m128 col = _mm_load_ps(&m2[i][0]);

        __m128 x = _mm_shuffle_ps(col,col, _MM_SHUFFLE(0, 0, 0, 0));
        __m128 y = _mm_shuffle_ps(col,col, _MM_SHUFFLE(1, 1, 1, 1));
        __m128 z = _mm_shuffle_ps(col,col, _MM_SHUFFLE(2, 2, 2, 2));
        __m128 w = _mm_shuffle_ps(col,col, _MM_SHUFFLE(3, 3, 3, 3));

        //dot products
        __m128 result_col = _mm_mul_ps(a, x);
        result_col = _mm_add_ps(result_col, _mm_mul_ps(b, y));
        result_col = _mm_add_ps(result_col, _mm_mul_ps(c, z));
        result_col = _mm_add_ps(result_col, _mm_mul_ps(d, w));

        _mm_store_ps(&out[i][0], result_col);
    }
}
```

There are actually a fair few ways of doing a 4x4 matmul, as it can change depending on how your matrix is laid out in memory. This version works for how GLM does things, and it's similar to how they themselves vectorize it.

# Checking the assembly
With Godbolt, we can see what is actually going on when the compiler deals with our code. Open this link to the code from the article [HERE](https://godbolt.org/z/n7WGnMnKa). Compiling both the array add and the matmul using the latest Clang with optimizations enabled, we can see a few interesting things.

The array additions have been autovectorized. The base version and the 4-unroll version have almost the exact same assembly code generated. The compiler is adding a check at the start of the function to see if the arrays are overlapping, and managed to vectorize the code by itself. On the version we vectorized ourselves, it's still unrolling the loop.

On the other hand, this is not the case with the matrix multiply. The compiler has completely failed to see through the code, and instead has decided to unroll the 2 inner loops of the matrix multiply. On the SIMD version, it has unrolled the outer loop, making it so the function doesn't loop at all.

When writing SIMD intrinsics, you generally want to check what code is being generated by the compiler, and in particular, always compare and benchmark it against the base version. In many cases, you can find that your manually written intrinsics end up running worse than what the compiler can do by itself, but also the compiler can fail to expand the operations and generate suboptimal code vs. naive intrinsics. 

Let's look at the benchmarks. This link [HERE](https://godbolt.org/z/jxb3nqnr9) is another version of the same code, but it has AVX enabled in compiler settings, and it has benchmarking you can look at on the right. At the time of writing this article, the timing looks like this. If you re-run it, the values might change as the Godbolt benchmarking can run on different cores, different clocks, and different machines.

```
--------------------------------------------------------
Benchmark              Time             CPU   Iterations
--------------------------------------------------------
add_scalar           559 ns          557 ns      1276163
add_sse             2284 ns          939 ns       741139
add_avx             1817 ns         1056 ns       657686
matmul_scalar       2.05 ns         1.22 ns    604972588
matmul_vector       1.62 ns        0.665 ns   1037437903
```

So we see something interesting. Turns out the autovectorization of the addition loop is almost 2x faster than our intrinsics version. Both the SSE version and AVX one. This is why it's important to measure and keep track of things, as our work here on the addition loop has proven itself to be useless, as the compiler can optimize things better here. 

On the matmul, we see the opposite. Our SIMD version is 2x faster vs. the scalar version. This is mostly what we normally would expect from an algorithm like this. While we are doing math 4-wide, there is overhead on moving data from SIMD registers and back from them. Meanwhile the CPU has multiple execution ports for simple float additions and multiplications, so it's already auto-parallelizing the code by itself. Normally it's rare to reach real 4x faster code, but 2 to 3x faster is common. If you do full 8-wide things with AVX you can reach 6x speedups against scalar.

For these benchmarks, I've enabled the `-mavx` flag in the compiler, which lets the compiler set AVX as the "default" level for the binaries. This made it generate better autovectorized code already using 8-wide SIMD operations, but it has done something very interesting to our matrix multiplication code. It has thrown out the shuffles, and moved the entire thing to use broadcast instructions instead. Essentially, it has converted the matrix multiplication into an "AVX" native version, which would look like this.


```cpp
void matmul4x4_avx(const mat4x4& m1, const mat4x4& m2, mat4x4& out){
    //load the entire m1 matrix to SIMD registers
    __m128  a = _mm_load_ps(&m1[0][0]);
    __m128  b = _mm_load_ps(&m1[1][0]);
    __m128  c = _mm_load_ps(&m1[2][0]);
    __m128  d = _mm_load_ps(&m1[3][0]);

     for (int i = 0; i < 4; i++) {

        __m128 x = _mm_broadcast_ss(&m2[i][0]);
        __m128 y = _mm_broadcast_ss(&m2[i][1]);
        __m128 z = _mm_broadcast_ss(&m2[i][2]);
        __m128 w = _mm_broadcast_ss(&m2[i][3]);

        //dot products
        __m128 result_col = _mm_mul_ps(a, x);
        result_col = _mm_add_ps(result_col, _mm_mul_ps(b, y));
        result_col = _mm_add_ps(result_col, _mm_mul_ps(c, z));
        result_col = _mm_add_ps(result_col, _mm_mul_ps(d, w));

        _mm_store_ps(&out[i][0], result_col);
    }
}
```

While AVX is about 8-wide SIMD, it brings new things to 4-wide vectors too, and one of the new things you get vs. SSE is the ability to load 1 value directly into a vector through the `_mm_broadcast_ss` instruction. So the compiler has found that we are doing a code pattern typical of SSE code to load 1 value to a whole vector, and replaced our intrinsics with the faster version. It's important to remember that the compiler will optimize around our intrinsics; it will not do 1-to-1 transformation of the intrinsics code into assembly. If we want that sort of reliability, there is no option but to directly use assembly code. 


# Frustum culling
Let's look into another algorithm commonly seen in graphics engines: frustum culling. We will be basing it on the one in [Learn OpenGL](https://learnopengl.com/Guest-Articles/2021/Scene/Frustum-Culling), but converting it to SIMD. You can read the article to understand what exact math operations we are dealing with.

For the frustum culling algorithm, we need to test 6 planes against a sphere per object. We have 2 main possibilities here. We could SIMD it in a horizontal fashion, culling the set of 6 planes against 8 objects at a time. We could also do it vertically, and do one object at a time, but calculating the 6 planes at once. Unfortunately, having 6 planes means we have a leftover if we do 4-wide, and it's too small for 8-wide. For this algorithm, people have done both versions depending on the game engine and CPU. We are going to implement the horizontal approach, parallelizing across the objects and culling 8 of them at a time, to demonstrate a few more advanced techniques.

Let's look at what exact code we are dealing with, in the scalar version. I've adapted the code from the Learn OpenGL article a bit to shorten it.

```cpp
struct Plane{
    glm::vec3 normal;
    float distance;
};

struct Frustum
{
    Plane faces[6];
};

struct Sphere{
    glm::vec3 center;
    float radius;

    bool isOnOrForwardPlane(const Plane& plane) const
    {
        return (glm::dot(plane.normal, center) - plane.distance) + radius > 0;
    }
};

bool isOnFrustum(const Frustum& camFrustum, const Sphere& sphere)
{
    
    for(int i = 0 ; i < 6; i++ )
    {
        if(!sphere.isOnOrForwardPlane(camFrustum.faces[i]))
        {
            return false;
        } 
    }

    return true;
}
```

We are skipping the sphere transformation code in the Learn OpenGL culling function as we will be assuming the spheres are already in their correct world transform to begin with, which skips a bunch of logic and lets us focus more on the core culling.

Looking at our data, we have Plane as essentially a vec4, and so is the sphere. A frustum is 6 planes, and we need to do dot product + a comparison to see on which side of the plane we are with the sphere. Then if the sphere is on the wrong side of any of the 6 planes, we return false on the cull function. 

Above, I mentioned that we are going to be parallelizing across multiple spheres at once. We see a problem here, which is how to deal with the branch. After all, we could have a case where one of the spheres is culled in the first iteration of the loop, while the others will be visible and thus go through the 6 iterations. 

To start with, we are going full branchless, and we will just not branch, instead we will keep the cull result in a vector and always loop 6 times.

We have another problem too. We don't have a clear axis to pack data for the vectors. Unlike with the matrix, we can't just take a row as a vector. For that, we are going to modify the algo to a Structure of Arrays layout. Instead of having each sphere as 1 struct, we will use 1 struct to hold 8 spheres. This way we have our clear vectors for all the logic. 

```cpp
//group 8 spheres in SoA layout, aligned to 32-byte boundaries so that it's well aligned for AVX 
struct alignas(32) SpherePack{

    float center_x[8];
    float center_y[8];
    float center_z[8];
    float radius[8];
};
```

If we want to do SIMD frustum culling, we can no longer just put the sphere in a render-mesh struct or similar. For this we need to look into data-oriented techniques, and we will need to store the cull spheres in a separate array, using packing like this. For example, it could look like this:
```cpp

struct CullGroup{

    std::vector<SpherePack> CullSpheres;
    std::vector<RenderMesh*> Renderables;
};
```

We can then index a specific renderable by dividing the index by 8 and accessing the sphere pack. This sort of memory layout is normally called AoSoA "Array of Structure of Arrays" and it tends to be highly effective. It would be fine too to store the 4 params from the sphere as full-length vectors; this is more for illustrational reasons.

We will be creating a new cull function, where we send it the float* pointers of the spheres, and we cull 8 at a time, returning a single u8 bitfield. In that bitfield, a set bit means true, and an unset bit means false. We are packing 8 bools into that number.

```cpp

uint8_t isOnFrustum(const Frustum& camFrustum, const SpherePack& spheres)
{
    uint8_t visibility = 0xFF; // begin with it set to true.
    for(int i = 0 ; i < 6; i++ )
    {
        visibility &= spheres.isOnOrForwardPlane(camFrustum.faces[i]);
    }

    return visibility;
}
```

We will be adding a function to the 8-pack of spheres that will compare the spheres with a single plane. This is what we will be writing intrinsics into. In here we could have an SSE version that culls 4 spheres 2 times, but we will continue with the 8-wide AVX.

Let's first unwrap the operations in the scalar cull so we can more clearly see the exact math we need.

```cpp
bool isOnOrForwardPlane(const Plane& plane) const
{
    float dx = center.x * plane.normal.x;
    float dy = center.y * plane.normal.y;
    float dz = center.z * plane.normal.z;

    float dot_distance = dx + dy + dz - plane.distance;

    return dot_distance > -radius;
}

```

And now we can convert it to the relevant intrinsics.

```cpp
struct alignas(32) SpherePack{

    float center_x[8];
    float center_y[8];
    float center_z[8];
    float radius[8];

    uint8_t isOnOrForwardPlane(const Plane& plane) const
    {
        __m256 cx = _mm256_load_ps(center_x);
        __m256 cy = _mm256_load_ps(center_y);
        __m256 cz = _mm256_load_ps(center_z);


        __m256 dx = _mm256_mul_ps(cx, _mm256_broadcast_ss(&plane.normal.x));
        __m256 dy = _mm256_mul_ps(cy, _mm256_broadcast_ss(&plane.normal.y));
        __m256 dz = _mm256_mul_ps(cz, _mm256_broadcast_ss(&plane.normal.z));

        __m256 dot_distance = _mm256_sub_ps( _mm256_add_ps(dx, _mm256_add_ps(dy,dz)) , _mm256_broadcast_ss(&plane.distance)) ;

        __m256 negrad = _mm256_sub_ps(_mm256_setzero_ps(), _mm256_load_ps(radius));
        __m256 comp = _mm256_cmp_ps(dot_distance, negrad , _CMP_GT_OQ);

        return _mm256_movemask_ps(comp);
    }
}
```

We begin by calculating the dot product and plane distance, and then we can compare it with the sphere radius negated. With AVX, you compare float values with the `_mm256_cmp_ps`, the `_CMP_GT_OQ` flag means its going to do a Greater-Than comparison. This will give you a special type of vector where it's going to be a mask. Other AVX operations will take this mask, such as the `_mm256_movemask_ps` we use to convert it to an integer. 

During our algorithm loop we will be repeating the same loads on the sphere on each of the 6 iterations of the loop. Let's hope the compiler can optimize that out as it would save perf.

We now have our wide frustum cull function, which culls 8 spheres at a time. 

There is still the issue of the branching in the loop, as looping 6 times can be unnecessary. We can add this branch to the loop to exit out of it. Under benchmark testing, this was a wash if it improved perf or not, as the branch is unpredictable and we are taking data "out" of the vector registers which has a bit of latency. The cull function itself is quite fast, so the latency of grabbing the mask and checking it in the branch can be larger than the cost of just doing the next loop of the cull. Unrolling this loop and interleaving operations would help, but it complicates the code a lot. This is a change of the algorithm that has to be benchmarked on the individual target CPUs to see if it's a win or not.

```cpp

uint8_t isOnFrustum(const Frustum& camFrustum, const SpherePack& spheres)
{
    uint8_t visibility = 0xFF; // begin with it set to true.
    for(int i = 0 ; i < 6; i++ )
    {
        visibility &= spheres.isOnOrForwardPlane(camFrustum.faces[i]);

        if(visibility == 0)
        {
            return visibility; // stop here
        }
    }

    return visibility;
}
```

This demonstrates culling operations wide, 8 items at a time. You can also try to change the algorithm to swap the axis of vectorization, by culling the 6 planes at a time vs. one sphere. You can do that as 4-wide planes first, then 2 scalar ones. Or pad the planes to 8. That's left as an exercise, try to do it on your own. The math and dot products are basically the same as with this version. 

# FMA (Floating Multiply-Add)

There is still one last thing to demonstrate here. As you have seen, dot products are a very common operation in graphics. Matrix mul uses them, and so does frustum culling. When doing dot products we are doing both multiplies and adds.

In some CPUs, there is support for the FMA extra instructions, which let you do multiply and add in one single operation. Not all CPUs support it, but when they do support it, it's pretty much 2x faster, as you are doing 1 operation to do both add and multiply vs. 2 operations, and if you look at the instruction performance tables, the cost of a multiply-add is the same as a single multiply, so it's essentially free perf. The operation is `(a * b) + c`. It does the multiply first, and then adds. There are variants where it subtracts.

Let's look at how we can use FMA to make things faster. Keep in mind only some CPUs have them, like Ryzens and some modern Intel CPUs, but the support is less widespread than AVX2 by itself. As FMA is a separate feature flag, there are AVX1-only CPUs that have FMA, while also AVX2 CPUs that don't. You must check before using it.

For the matrix multiply, if we move it to use FMA, it looks like this.

```cpp
void matmul4x4_avx_fma(const mat4x4& m1, const mat4x4& m2, mat4x4& out){
    //load the entire m1 matrix to SIMD registers
    __m128  a = _mm_load_ps(&m1[0][0]);
    __m128  b = _mm_load_ps(&m1[1][0]);
    __m128  c = _mm_load_ps(&m1[2][0]);
    __m128  d = _mm_load_ps(&m1[3][0]);

     for (int i = 0; i < 4; i++) {

        __m128 x = _mm_broadcast_ss(&m2[i][0]);
        __m128 y = _mm_broadcast_ss(&m2[i][1]);
        __m128 z = _mm_broadcast_ss(&m2[i][2]);
        __m128 w = _mm_broadcast_ss(&m2[i][3]);

        // dot products
        __m128 resA = _mm_fmadd_ps(x, a, _mm_mul_ps(y, b));
        __m128 resB = _mm_fmadd_ps(z, c, _mm_mul_ps(w, d));

        _mm_store_ps(&out[i][0],  _mm_add_ps(resA, resB));
    }
}
```

This version should give us extra performance. On my computer it shows 20% extra performance against the normal AVX version, which is a very nice performance win.

For the culling, the FMA version looks like this.

```cpp
uint8_t isOnOrForwardPlane_fma(const Plane& plane) const
{
    __m256 cx = _mm256_load_ps(center_x);
    __m256 cy = _mm256_load_ps(center_y);
    __m256 cz = _mm256_load_ps(center_z);
    __m256 vradius = _mm256_load_ps(radius);

    __m256 px = _mm256_broadcast_ss(&plane.normal.x);
    __m256 py = _mm256_broadcast_ss(&plane.normal.y);
    __m256 pz = _mm256_broadcast_ss(&plane.normal.z);
    __m256 pdist = _mm256_broadcast_ss(&plane.distance);

    // (center.x * plane.normal.x) + (center.y * plane.normal.y)
    __m256 dot_a = _mm256_fmadd_ps(cx, px,
                    _mm256_mul_ps(cy, py));

     // (center.z * plane.normal.z) - plane.distance
    __m256 dot_b = _mm256_fmsub_ps(cz,pz, pdist);

    __m256 dot_distance = _mm256_add_ps(dot_a,dot_b);

    __m256 comp = _mm256_cmp_ps(dot_distance, _mm256_sub_ps(_mm256_setzero_ps(), vradius), _CMP_GT_OQ);

    return _mm256_movemask_ps(comp);
}
```

FMA can often be hard to deal with as you are doing multiple operations in 1 function call, which complicates the code. Always benchmark it to see if you get a win. On my testing of this frustum cull function, it can improve the performance up to 40% vs. normal AVX version, so the win is often very worth it. FMA is not something the compiler will optimize your intrinsics into, as FMA has different floating-point properties (it's higher precision!) and rounding vs. multiply and add as normal.

{% include comments.html term="Intro To SIMD Comments" %}