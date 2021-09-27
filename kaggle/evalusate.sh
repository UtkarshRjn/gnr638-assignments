#!/bin/bash

diff -w $1 label.csv | grep "^>" | wc -l