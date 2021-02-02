# general
Code base for the [Acoustics Lab in BGU](https://sites.google.com/view/acousticslab).

## Usage
Run `startup_script()` to append the required folders to the path,
as well as to set some recommended graphics defaults.

See the `+examples` folder for usage examples.

The code is written for MATLAB 2020a and above.

## How to Contribute
Your modification should always be on your own branch.
If you fill they are ready, create a pull request to merge your branch with `main`.

General function should go into the folder `general`.
Specific projects should be in their own package
(a folder whose name starts with a `+`, read more
[here](https://www.mathworks.com/help/matlab/matlab_oop/scoping-classes-with-packages.html)).

If your package uses a function that is too specific to be in the `general` folder, 
the function file should be in a folder named `private`, under the package folder.
This will ensure that this function is visible only to code inside the package 
(read more [here](https://www.mathworks.com/help/matlab/matlab_prog/private-functions.html)).

Each package should contain a `README.md` file that:
* Explains the package interface and how to use it.
* Contains the name of the main authors.
* Links to relevant references (if available).

You can learn more about markdown (md) file 
[here](https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet).