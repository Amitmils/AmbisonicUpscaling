# ToolBoxApp
A matlab app for various signal processing algorithms.

## usage

* Clone this repository using github desktop or run this command:
`git clone https://github.com/ACLab-BGU/general.git`

* Open the folder using matlab
* Open the ToolboxApp/toolbox.mlapp and press run
* Add source(s)
* Set your desired room parameters along with array and source(s) locations.
 ![image](https://user-images.githubusercontent.com/13310488/131481214-e41b58b4-f104-4a8f-8ac2-12aa2fe622c6.png)

* Select algorithm and parameters.
* ![image](https://user-images.githubusercontent.com/13310488/126484510-ca0e0910-6ae3-454f-a253-3aebf6029a4d.png)

* Proccess the signal and listen to the results




## contributing

For including your own algorithm in this app, follow this steps:
1.  create a file in the functions folder with a function that get the signal and other parameters as you wish.
2.  create a new processor class in the processor folder using the example.
3.  your processor should contain:
     1.  a "process" funtion that get signal, frequecy and room parameter and return the new signal.
     2.  all of your algorithm parameters inside of a dictionary (matlab's Map object) with the corresponding key. make shure to use valuable keys, 
        you'll need this later for binding your arguments and GUI component.
4.  In the toolbox.mlapp add a tab for the tab window using the '+' sign and add your gui component accoring to the desired parameters.
     make shure to use valuable names for each components.
     
![image](https://user-images.githubusercontent.com/13310488/111609309-d45aa100-87e2-11eb-8a22-624eb1230769.png)

5. add your tab to the tabs Map structure in the sturtUpFcn:
```
app.tabs('YOUR-ALGORITHM-NAME') = app.YourTab;
```

6. in the code view go to startUpfcn, in the processor section construct your processor and bind it to the GUI component. for example:
```
app.processor('YOUR-ALGORITHM-NAME') = yourprocessor();
app.processor('YOUR-ALGORITHM-NAME').bind(yourGuiComponent,'Your-arg-name');
app.processor('YOUR-ALGORITHM-NAME').bind(yourGuiComponent2,'Your-arg-name2','r'); %'r' for rounding values

```
