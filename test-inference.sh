#!/bin/bash

actual_output=$(zig build run -Doptimize=ReleaseFast -- stories260K.bin -z tok512.bin -t 0 -n 200 --test)

expected_output="Once upon a time, there was a little girl named Lily. She loved to play outside in the park. One day, she saw a big, red ball. She wanted to play with it, but it was too high.
Lily's mom said, \"Lily, let's go to the park.\" Lily was sad and didn't know what to do. She said, \"I want to play with your ball, but I can't find it.\"
Lily was sad and didn't know what to do. She said, \"I'm sorry, Lily. I didn't know what to do.\"
Lily didn't want to help her mom, so she"

if [ "$actual_output" == "$expected_output" ]; then
    echo "test ok"
else
    echo "test failed"
    exit 1
fi
