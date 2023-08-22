#!/bin/bash

set -e           # Terminates script at the first error
set -o pipefail  # Sets the exit status for pipes
set -u           # Triggers an error when an unset variable is called
set -o noclobber # Prevents from overwriting existing files

zig build

actual_output=$(./zig-out/bin/llama2 stories260K.bin -z tok512.bin -t 0 -n 200 --test)

expected_output="Once upon a time, there was a little girl named Lily. She loved to play outside in the park. One day, she saw a big, red ball. She wanted to play with it, but it was too high.
Lily's mom said, \"Lily, let's go to the park.\" Lily was sad and didn't know what to do. She said, \"I want to play with your ball, but I can't find it.\"
Lily was sad and didn't know what to do. She said, \"I'm sorry, Lily. I didn't know what to do.\"
Lily didn't want to help her mom, so she"

if [ "$actual_output" != "$expected_output" ]; then
    echo "argmax test failed"
    exit 1
fi

actual_output=$(./zig-out/bin/llama2 stories260K.bin -z tok512.bin -t 1 -p 1 -s 42 -n 200 --test)

expected_output="Once upon a time, there was a colorful little bird named Tweety. Tweet loved to chase the bird from a hat. One day, Tweetie touched and grabbed a subway chest. The cat looked at the chicken, but his together was finish and if he knew it was busy.
After they finished eating it, Tweetie went to the table. She found many treats and flew down. The chicken had both lepped on a bright out of the world."

if [ "$actual_output" != "$expected_output" ]; then
    echo "temperature test failed"
    exit 1
fi

actual_output=$(./zig-out/bin/llama2 stories260K.bin -z tok512.bin -t 1 -p 0.95 -s 42 -n 200 --test)

expected_output="One day, a boy named Tim found a big pot. He found a little girl named Sue. Tim loved to decorate the pot. He had a boy named Tim. Tim was looking for a dress with some food.
Tim and his mom asked, \"What's your doll, but I broke a box. The doctor said to your boy, but I won't tell him to get out.\" Tim went to the pot. He said, \"No, I can help you. It is my new dream.\"
Sue wanted to help the doll. She decided to play a game in"

if [ "$actual_output" != "$expected_output" ]; then
    echo "nucleus sampling test failed"
    exit 1
fi

echo "tests ok"
