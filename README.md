# llama2

a basic port of Andrej Karpathys [llama2.c](https://github.com/karpathy/llama2.c) to rust.

used as a learning project to familiarize myself with rust and llama2 architecture.

## running instruction

to build the app run
```
rustc -C opt-level=3 run.rs -o run.exe
```
then download any of the following three models

mac: 
```
wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin
wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories42M.bin
wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories110M.bin
```

windows:
```
Invoke-WebRequest -Uri "https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin" -OutFile "stories15M.bin"
Invoke-WebRequest -Uri "https://huggingface.co/karpathy/tinyllamas/resolve/main/stories42M.bin" -OutFile "stories42M.bin"
Invoke-WebRequest -Uri "https://huggingface.co/karpathy/tinyllamas/resolve/main/stories110M.bin" -OutFile "stories110M.bin"
```

then ```
./run stories15M.bin 0.9 256 "Once upon a time"```
to run at temperature=0.9, steps=256, prompt = "Once upon a time" using stories15M model.

can also just do ```./run stories15M.bin``` to run stories15M model with default params.
