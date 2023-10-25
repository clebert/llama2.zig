#!/bin/bash

set -e           # Terminates script at the first error
set -o pipefail  # Sets the exit status for pipes
set -u           # Triggers an error when an unset variable is called
set -o noclobber # Prevents from overwriting existing files

if [ "$#" -ne 2 ]; then
    echo "Usage: ./benchmark.sh <model> <runs>"
    exit 1
fi

model=$1
runs=$2

zig build -Doptimize=ReleaseFast

for ((worker_count=0; worker_count<11; worker_count++))
do
  echo -n "Running $model with $worker_count workers:"

  total=0
  declare -a scores

  for ((run=0; run<runs; run++))
  do
    output=$(./zig-out/bin/llama2-generator "$model" --temperature 0 --verbose --worker_count "$worker_count")
    line=$(echo "$output" | tail -n 1)
    score=$(echo "$line" | grep -o -E '[0-9]+' | head -1 | sed -e 's/^0\+//')
    scores+=("$score")
    total=$((total + score))
  done

  min=${scores[0]}
  max=${scores[0]}

  for score in "${scores[@]}"
  do
    if [[ "$score" -lt "$min" ]]
    then
      min="$score"
    fi

    if [[ "$score" -gt "$max" ]]
    then
      max="$score"
    fi
  done

  average=$((total / runs))
  minAvg=$((min-average))
  maxAvg=$((max-average))

  printf "\t%s\t%s\t+%s\n" "$average" "$minAvg" "$maxAvg"

  unset scores
done
