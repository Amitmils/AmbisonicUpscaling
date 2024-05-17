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


# How to Run the Binaural Reproduction Tutorial

1. **Sign In or Create an Account**: Visit [GitHub](https://github.com) and log in to your account. If you don't have an account, sign up for free.

2. **Accept the ACLab-BGU Invitation**: Once logged in, accept the invitation to join [ACLab-BGU](https://sites.google.com/view/acousticslab), a private group containing internal lab repositories and code. Contact berebio@post.bgu.ac.il to receive the invitation.

3. **Access the Repository**: Navigate to your repositories and locate the repository named "general." This repository includes the Binaural Reproduction tutorial, among other resources.

4. **Download and Install GitHub Desktop**: Install the GitHub Desktop client GUI, which allows you to clone, push/pull, and commit changes to ACLab-BGU's git repository. Download GitHub Desktop from [here](https://desktop.github.com).

5. **Clone the Repository**: After installing GitHub Desktop, go to the "general" repository and click the green "<> Code" button. Then, select "Open with GitHub Desktop" to clone the repository to your computer.

6. **Approve GitHub Desktop Prompts**: Approve any prompts that appear in GitHub Desktop. You'll now see the "general" repository listed in the left panel of the GitHub Desktop client. By default, the repository files are located in `/Users/YourUserName/Documents/GitHub/general/`.

7. **Run the Startup Script**: Navigate to the "general" folder on your computer. Locate the `startup_script.m` file and run it in MATLAB. Ensure that your current folder location is the "general" folder by right-clicking on the `startup_script.m` file in MATLAB and selecting "Change current Folder to /Users/../general/".

8. **Run the Binaural Reproduction Tutorial**: In the MATLAB file navigator, open and run `./general/+examples/pwd_binaural_reproduction.m`. This is the Binaural Reproduction tutorial. Related papers are listed alongside a brief description in the comments at the start of the script.

9. **Explore the Tutorial**: The script will generate a graphical representation of the simulated room, display the Impulse Response of the zero-order SH, and play audio examples of the resulting `p(t)` signals (binaural signal captured with the mic array, non-binaural signals captured by a single mic, dry anechoic signal).

10. **Explore the Code**: Feel free to explore the code further. However, avoid committing/pushing any changes back to the main repository branch unless you have a valuable feature to contribute.
