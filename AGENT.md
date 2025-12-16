# Instructions for AI Agent

This repository provides an easy interface to generate Random Dot Kinematogram (RDK) displays that can be used in psychophysical experiments. 
Specifically, it uses PsychoPy and custom functions to generate RDKs with many different parameters. The basic parameters are taken from psychopy.visual.DotStim, but the key here is that we add a Gaussian envelope on the RDK field. A PsychoPy implementation is provided in `rdk_generator/psychopy_adapter.py`. Here, we add to this functionality by distributing the color of the dots (black and white) such that the contrast of the entire field matches the background (i.e., mean luminance inside the field equals the grey background). 

Furthermore, this repo adds a user-friendly Flask webapp that can be used to change the parameters associated with the RDK and Gaussian filter and generate movies that can be previewed in browser, downloaded in `.mp4` format, or exported as frames to be displayed via PsychoPy/Psychtoolbox software in real experiments.

The agent's task is to:
    1. Write a plan for generating this repository according to best practices.
    2. Generate the scaffold and make markdown files that document what each module should do.
    3. Write Python code to implement these modules with excellent documentation. 
    4. Write and run comprehensive tests.
