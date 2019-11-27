#!/bin/bash
input_file=$1
dest_dir=$2
new_file=$3
mkdir -p "$dest_dir"
cp "$input_file" "${dest_dir}/${new_file}"
