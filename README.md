# InfiniTrain

## 🚀 Getting Started

### 🛠️ Build Instructions

```bash
mkdir build
cd build
cmake ..           # Use -DUSE_CUDA=ON to enable CUDA support
make
```

## 🧪 Running Examples

Each model in the `example/` directory is compiled into an independent executable.  
For instance, the `mnist` example will produce a binary named `mnist`.

You can view the available runtime options by executing:

```bash
./mnist --help
```