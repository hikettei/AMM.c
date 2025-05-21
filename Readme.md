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

### Project Design

(TODO: 英訳)

- [ ] 将来より良い近似アルゴリズムが出るかもしれない。その際になるべく何も考えないで実装を追加したい。各手法間で共通項があっても一才抽象化を実施しない。その代わりにEntry Pointを複数もつ。

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
