#!/bin/bash

echo "Strating program at $(date)" # get the date

echo "Running program $0 with $# arguments with pid $$"

for file in "$@"; do
	grep foobar "$file" > /dev/null 2> /dev/null
	# when pattern isn't found exit status is 1
	if [[ "$&" -ne 0 ]]; then
		echo "File $file doesn't have any foobar, adding one as a comment"
		echo "# foobar" >> "$file"
	fi
done

