# AMM.c

> AMM = Approximated Matrix Multiplication

### Motivation

- Dependency-Free
- Tiny
- Multi-Platform
- Simple API (callable via FFI)
- Fast
- The best platform for experiments
- Supporting Lowert Bits (2 ~ 8bits integers)

### Project Status

- [ ] FP32 Maddness
- [ ] (Experimental) INT2 ~ INT8 Encoder Training

### Project Design


- [ ] In the future, better approximation algorithms may emerge. When that happens, I want to be able to add implementations with as little thought as possible. Even if there are commonalities between methods, I will not perform any abstraction. Instead, I will provide multiple entry points.

### Supported Architectures

- [ ] arm64
- [ ] x64 (AVX2, AVX512)
- [ ] CUDA
- [ ] Metal


### Build

Requirements: Build, `cmake > 3.14`, Ninja

```sh
$ mkdir build && cd build
$ cmake ..
$ cmake --build .
```

### Testing

- Requirements: cunit

### TODO

- [ ] Implement Maddness
- [ ] Build FFI and `.dylib`
- [ ] Test on CI (FFI Call from Common Lisp or Python)
- [ ] Benchmark on CI
