# llama2

attempting to port Karpathys llama2.c to rust

## running instruction

first run '''rustc -C opt-level=3 run.rs -o run.exe''' to build app
download any of following models \n mac: \n '''wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin
wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories42M.bin
wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories110M.bin'''

windows: \n '''Invoke-WebRequest -Uri "https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin" -OutFile "stories15M.bin"
Invoke-WebRequest -Uri "https://huggingface.co/karpathy/tinyllamas/resolve/main/stories42M.bin" -OutFile "stories42M.bin"
Invoke-WebRequest -Uri "https://huggingface.co/karpathy/tinyllamas/resolve/main/stories110M.bin" -OutFile "stories110M.bin"'''

then '''./run stories15M.bin 0.9 256 "Once upon a time"''' to run at temperature=0.9, steps=256, prompt = "Once upon a time" using stories15M.bin model.

can also just do '''./run stories15M.bin''' to run model.
