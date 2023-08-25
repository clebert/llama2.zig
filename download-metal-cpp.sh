#!/bin/bash

set -e           # Terminates script at the first error
set -o pipefail  # Sets the exit status for pipes
set -u           # Triggers an error when an unset variable is called
set -o noclobber # Prevents from overwriting existing files

OS_VERSION_FULL=$(sw_vers -productVersion)

if [[ $OS_VERSION_FULL =~ ([0-9]+)\.([0-9]+)(\.([0-9]+))? ]]; then
    OS_MAJOR_VERSION=${BASH_REMATCH[1]}
    OS_MINOR_VERSION=${BASH_REMATCH[2]}
fi

OS_VERSION="$OS_MAJOR_VERSION.$OS_MINOR_VERSION"

if [[ $OS_MAJOR_VERSION == "12" ]]
then
    ZIP_FILE_NAME="metal-cpp_macOS12_iOS15.zip"
elif [[ $OS_MAJOR_VERSION == "13" ]]
then
    if [[ $OS_MINOR_VERSION -le 2 ]]
    then
        ZIP_FILE_NAME="metal-cpp_macOS13_iOS16.zip"
    else
        ZIP_FILE_NAME="metal-cpp_macOS13.3_iOS16.4.zip"
    fi
elif [[ $OS_MAJOR_VERSION == "14" ]]
then
    ZIP_FILE_NAME="metal-cpp_macOS14_iOS17-beta.zip"
else
    echo "Unsupported macOS version: $OS_VERSION"
    exit 1
fi

curl -LO https://developer.apple.com/metal/cpp/files/$ZIP_FILE_NAME
unzip $ZIP_FILE_NAME -d .
rm $ZIP_FILE_NAME
