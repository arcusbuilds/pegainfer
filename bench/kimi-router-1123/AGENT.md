# Task: benchmark the Kimi-K2 router kernel on an RTX 5070 Ti

You are helping test a pegainfer pull request
(https://github.com/pegainfer-project/pegainfer/pull/1170). Everything you need
is in this folder. Please follow these steps and report back.

## Rules

- Only build and run `router_5070.cu`. Don't run any other script in this repo.
- Don't edit `router_5070.cu`. If it doesn't build, report the exact error
  instead of changing the code.
- Don't push, open PRs, or comment on GitHub. Send the results to the person
  who asked you.

## What the file does

`router_5070.cu` holds two versions of one CUDA kernel (old and new). It runs
both on the same inputs, checks the outputs match byte for byte (3,554 cases),
then times both at batch sizes 1 to 141. It prints everything to stdout and
ends with `RESULT: PASS` or `RESULT: FAIL`.

## Steps

1. Record the setup:
   - `nvidia-smi`
   - `nvcc --version` (it must be CUDA 12.8 or newer, older versions can't build for sm_120)
2. Make sure the GPU is idle: close games, browsers with video, and other GPU
   jobs. `nvidia-smi` should show no other heavy process.
3. Build. On Windows, use "x64 Native Tools Command Prompt for VS" so nvcc can
   find the C++ compiler:
   ```
   nvcc -O3 -arch=sm_120 router_5070.cu -o router_5070
   ```
   The build prints 4 "declared but never referenced" warnings. They are
   expected.
4. Run it 3 times, saving each run:
   ```
   ./router_5070 > result1.txt
   ./router_5070 > result2.txt
   ./router_5070 > result3.txt
   ```
   (On Windows use `.\router_5070.exe`.) Each run takes about 30 seconds.

## What to send back

1. The full text of `result1.txt`, `result2.txt` and `result3.txt`.
2. The `nvidia-smi` and `nvcc --version` output from step 1.
3. One line on anything unusual: build errors, a FAIL, or timings that differ a
   lot between the 3 runs.

## What we expect (so you can sanity-check)

The 5070 Ti has 70 SMs. The new kernel should be faster than the old one up to
batch size 70, then slower from 71, with another step at 141. If the results
look different from that, report them as they are. Don't rerun until they
match.
