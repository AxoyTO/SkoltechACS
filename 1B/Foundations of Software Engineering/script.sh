#!/bin/sh

echo "Hello, world!";
echo "Appended line";

output=$(ls *sh)
echo "$(output)"

for i in $(seq 1 2 5)
do
	mkdir -p ${i}folder
	echo "Created ${i}folder"
done

echo "Listing:"
echo "$(ls -la *.sh)"

for i in $(seq 1 2 5)
do
	rmdir -p ${i}folder
	echo "Removed ${i}folder"
done

# function
fse_mcd()
{
	mkdir -p $1folder
	rmdir $1folder
}

for i in $(seq 1 2 5)
do
	#mkdir -p ${i}folder
	fse_mcd $i
done
