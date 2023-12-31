#!/bin/bash

set -e           # Terminates script at the first error
set -o pipefail  # Sets the exit status for pipes
set -u           # Triggers an error when an unset variable is called
set -o noclobber # Prevents from overwriting existing files

zig build

model_path="models/tinystories_260k"

actual_output=$(./zig-out/bin/llama2-generator $model_path --temperature 0 --sequence_length 200 --worker_count 0)

# Generated with llama2.c (https://github.com/karpathy/llama2.c/tree/7ac65cb2c2b169050747be92011b7bebdd1b4544)
expected_output="Once upon a time, there was a little girl named Lily. She loved to play outside in the park. One day, she saw a big, red ball. She wanted to play with it, but it was too high.
Lily's mom said, \"Lily, let's go to the park.\" Lily was sad and didn't know what to do. She said, \"I want to play with your ball, but I can't find it.\"
Lily was sad and didn't know what to do. She said, \"I'm sorry, Lily. I didn't know what to do.\"
Lily didn't want to help her mom, so she"

if [ "$actual_output" != "$expected_output" ]; then
    echo "argmax test failed"
    exit 1
fi

actual_output=$(./zig-out/bin/llama2-generator $model_path --top_p 1 --random_seed 42 --sequence_length 200 --worker_count 0)

# Generated with llama2.c (https://github.com/karpathy/llama2.c/tree/7ac65cb2c2b169050747be92011b7bebdd1b4544)
expected_output="Once upon a time, there was a big roof. The fox was ready to look for people inside. He saw a big rock near a big tree. The roof was very small and fun! He ate the roof too. He got a shiny stool, so he sicked the roof with his friend, the girl named Mia.
\"Help, Mia. Why are you sad, Mia?\" she asked.
\"I want to try us,\" Mia said. \"It is cute. We have to find one of smells!\"
Mia felt proud of"

if [ "$actual_output" != "$expected_output" ]; then
    echo "temperature test failed"
    exit 1
fi

actual_output=$(./zig-out/bin/llama2-generator $model_path --top_p 0.95 --random_seed 42 --sequence_length 200 --worker_count 0)

# Generated with llama2.c (https://github.com/karpathy/llama2.c/tree/7ac65cb2c2b169050747be92011b7bebdd1b4544)
expected_output="Once upon a time, there was a little boy named Timmy. Timmy loved going to the park with his mom. One day, Lily went outside to play outside in her pocket. He was scared and didn't know where to buy some colorful animals.
Later that day, Timmy's mom came outside and saw Timmy just playing in the shore. She didn't see that made him happy, but Timmy's mom said he had to sort his cobwebs and write cold colors. Timmy asked his mom if he was amazed! His mom"

if [ "$actual_output" != "$expected_output" ]; then
    echo "nucleus sampling test failed"
    exit 1
fi

actual_output=$(./zig-out/bin/llama2-generator $model_path --top_p 0.95 --random_seed 42 --sequence_length 200 --prompt "There was a big" --worker_count 0)

# Generated with llama2.c (https://github.com/karpathy/llama2.c/tree/7ac65cb2c2b169050747be92011b7bebdd1b4544)
expected_output="There was a big pretty grass. It was a long elephant. The cars wanted to tell him that as they spin before the amazing doll, just like it she was always okay.
One day, a little girl named Lucy found a ball and an axe. She wanted to race, but she didn't know what the ball was. The ball was determined to be a nice car for Lucy.
Lucy's friend, a little girl named Lily, said, \"Let's go to the store with my mom. I will s"

if [ "$actual_output" != "$expected_output" ]; then
    echo "input prompt test failed"
    exit 1
fi

actual_output=$(./zig-out/bin/llama2-generator $model_path --temperature 0 --sequence_length 200 --worker_count 3)

# Generated with llama2.c (https://github.com/karpathy/llama2.c/tree/7ac65cb2c2b169050747be92011b7bebdd1b4544)
expected_output="Once upon a time, there was a little girl named Lily. She loved to play outside in the park. One day, she saw a big, red ball. She wanted to play with it, but it was too high.
Lily's mom said, \"Lily, let's go to the park.\" Lily was sad and didn't know what to do. She said, \"I want to play with your ball, but I can't find it.\"
Lily was sad and didn't know what to do. She said, \"I'm sorry, Lily. I didn't know what to do.\"
Lily didn't want to help her mom, so she"

if [ "$actual_output" != "$expected_output" ]; then
    echo "multithreaded test failed"
    exit 1
fi

echo "tests ok"
