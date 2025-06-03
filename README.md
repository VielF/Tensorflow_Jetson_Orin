# Tensorflow_Jetson_Orin

Tutorial repository to compile Tensorflow in NVIDIA Jetson Nano with Jetpack 6.0, CUDA 12.2, cuDNN 8.9, and TensorRT 8.6 to use Tensorflow C API.


---

## 📋 Requirements

- **Jetson board** (e.g., Orin Nano)
- **CUDA 12.2**
- **cuDNN 8.9**
- **TensorRT 8.6**
- **Python 3.10**
- **Bazel** (>= 6.x) - Used 6.5
- **SWAP** enabled (recommended: at least 40GB)

---

## 📋 Creating the absl_neon.patch to NEON and NVCC to compile withouts errors

### 📥 Clone the correct version of Abseil used by TensorFlow

Check the commit hash in the workspace.bzl (inside /tensorflow/third_party/absl/). For example:

ABSL_COMMIT = "fb3621f4f897824c0dbe0615fa94543df6192f30"

So clone it:

```bash
git clone https://github.com/abseil/abseil-cpp.git
cd abseil-cpp
git checkout fb3621f4f897824c0dbe0615fa94543df6192f30
```

### ⚙️Modify the file

Edit absl/base/config.h and replace:

```c
#elif defined(__ARM_NEON) && !defined(__CUDA_ARCH__)
```

with:

```c
#elif defined(__ARM_NEON) && !defined(__CUDACC__)
```

(or vice versa, depending on your needs)

### 🔨Generate the patch

In the abseil-cpp directory:

```bash
git diff > ~/tensorflow/third_party/absl/absl_neon.patch
```

This ensures a valid unified diff format.

### 🔁Double-check the patch

View the start of the file:

```bash
head -n 15 ~/tensorflow/third_party/absl/absl_neon.patch
```

It should look like this:

```bash
diff
Copy
Edit
diff --git a/absl/base/config.h b/absl/base/config.h
index 5fa9f0e..741e320 100644
--- a/absl/base/config.h
+++ b/absl/base/config.h
@@ -962,7 +962,7 @@ static_assert(ABSL_INTERNAL_INLINE_NAMESPACE_STR[0] != 'h' ||
 // https://llvm.org/docs/CompileCudaWithLLVM.html#detecting-clang-vs-nvcc-from-code
 #ifdef ABSL_INTERNAL_HAVE_ARM_NEON
 #error ABSL_INTERNAL_HAVE_ARM_NEON cannot be directly set
-#elif defined(__ARM_NEON) && !defined(__CUDA_ARCH__)
+#elif defined(__ARM_NEON) && !defined(__CUDACC__)
 #define ABSL_INTERNAL_HAVE_ARM_NEON 1
 #endif
```

### ⚙️Add the patch file referecen in /tensorflow/third_party/absl/workspace.blz

```bash
    tf_http_archive(
        name = "com_google_absl",
        sha256 = ABSL_SHA256,
        build_file = "//third_party/absl:com_google_absl.BUILD",
        system_build_file = "//third_party/absl:system.BUILD",
        system_link_files = SYS_LINKS,
 ->     patch_file = ["//third_party/absl:absl_neon.patch"],
        strip_prefix = "abseil-cpp-{commit}".format(commit = ABSL_COMMIT),
        urls = tf_mirror_urls("https://github.com/abseil/abseil-cpp/archive/{commit}.tar.gz".format(commit = ABSL_COMMIT)),
    )

```

## 🧠 How to Increase SWAP on Jetson Orin Nano (Ubuntu 22.04)

Follow these steps to increase SWAP memory on your Jetson Orin Nano. This is especially useful when compiling large projects like TensorFlow.

---

### 1️⃣ Check Current SWAP

```bash
swapon --show
free -h
```

---

### 2️⃣ Create a New SWAP File (e.g., 4 GB)

> You can replace `4G` with `2G`, `8G`, etc., depending on available disk space.

```bash
sudo fallocate -l 4G /swapfile_extra
```

If `fallocate` doesn't work, use:

```bash
sudo dd if=/dev/zero of=/swapfile_extra bs=1M count=4096
```

---

### 3️⃣ Set Correct Permissions

```bash
sudo chmod 600 /swapfile_extra
```

---

### 4️⃣ Configure as SWAP

```bash
sudo mkswap /swapfile_extra
```

---

### 5️⃣ Activate the New SWAP Area

```bash
sudo swapon /swapfile_extra
```

---

### 6️⃣ Confirm It’s Active

```bash
swapon --show
free -h
```

---

### 7️⃣ (Optional) Make It Permanent After Reboot

Add the following line to the end of `/etc/fstab`:

```bash
echo '/swapfile_extra none swap sw 0 0' | sudo tee -a /etc/fstab
```

---

## 📋 Compiling the Tensorflow v2.16.x

### 📥 Clone TensorFlow

```bash
git clone https://github.com/tensorflow/tensorflow
cd tensorflow
```

---

### 🔁 Checkout a Compatible Version

> ⚠️ TensorFlow 2.18+ does **not** compile on Jetson due to NCCL requirement (unsupported). Using v2.16.x or v2.15.x.

```bash
git checkout v2.16.1  # Or v2.15.x
```

---

### ⚙️ Configure Build

Run the configuration script and answer the prompts:

```bash
./configure
```

Answers:

- CUDA support → `Y`
- TensorRT support → `Y`
- CUDA compute capability → `8.7`
- Use clang as CUDA compiler → `N`

This generates `.tf_configure.bazelrc`.

---

### 🐍 Set Python Version

```bash
export TF_PYTHON_VERSION=3.10
```

---

### 🧹 Clean Build Cache (Important)

```bash
bazel clean --expunge
```

---

### 🩹 Apply `absl_neon.patch`

> Required to fix ARM NEON vs CUDA conflict on Jetson. Follow official patching steps:

---

### 🔨 Build TensorFlow C API (Static Lib)

```bash
bazel build -c opt \
  --config=opt \
  --verbose_failures \
  --config=noaws \
  --config=nogcp \
  --config=nohdfs \
  --config=cuda \
  --define=using_cuda=true \
  --define=using_cuda_nvcc=true \
  --config=nonccl \
  --action_env=TF_CUDA_VERSION=12.2 \
  --action_env=CUDA_TOOLKIT_PATH="/usr/local/cuda-12.2" \
  --action_env=CUDNN_INSTALL_PATH="/usr/lib/aarch64-linux-gnu" \
  --action_env=LD_LIBRARY_PATH="/usr/local/cuda-12.2/lib64:/usr/lib/aarch64-linux-gnu" \
  //tensorflow/tools/lib_package:libtensorflow
```

---

### 📦 Output

After compilation, you'll find the package here:

```
bazel-bin/tensorflow/tools/lib_package/libtensorflow.tar.gz
```

This archive contains:

- `libtensorflow.so`
- `libtensorflow_framework.so`
- Headers for C API

---

### 📦 Install (Optional)

```bash
sudo tar -C /usr/local -xzf bazel-bin/tensorflow/tools/lib_package/libtensorflow.tar.gz
sudo ldconfig /usr/local/lib
```
