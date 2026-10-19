pegainfer #1123 router kernel check - RTX 5070 Ti kit
=====================================================

What this is
------------
One CUDA file, router_5070.cu. It runs the old and the new version of one
GPU kernel on the same inputs. It checks that both give exactly the same
output bits, then times both. No Python, no other files needed.

What you need
-------------
- An NVIDIA RTX 5070 Ti (any Blackwell card works).
- CUDA Toolkit 12.8 or newer (older versions cannot build for this GPU).
- Windows: Visual Studio with the C++ tools (for the compiler nvcc uses).

Windows
-------
1. Open "x64 Native Tools Command Prompt for VS" from the Start menu.
2. cd to the folder with router_5070.cu.
3. Build:
     nvcc -O3 -arch=sm_120 router_5070.cu -o router_5070.exe
4. Run:
     .\router_5070.exe > result.txt

Linux
-----
     nvcc -O3 -arch=sm_120 router_5070.cu -o router_5070
     ./router_5070 > result.txt

Notes
-----
- The build prints 4 warnings like "variable kKimiExperts was declared but
  never referenced". That is expected. Ignore them.
- The run takes about 30 seconds. The screen stays empty while it runs,
  because all output goes into result.txt.
- Close games or other GPU-heavy apps first, so the timings are clean.
- The last line of result.txt says RESULT: PASS or RESULT: FAIL.

Please send back result.txt. That's all.
