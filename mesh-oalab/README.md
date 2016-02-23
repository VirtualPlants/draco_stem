# Mesh - OALab

A set of plugins and components for the best mesh experience in OpenAleaLab (and TissueLab)

## Contents

### The TopomeshControls service (currently an applet) 

Add the applet to your OALab environment as any regular applet :
* In a workspace right click and select "Edit Layout"
* Add a new tab (right click + "Add Tab") if necessary
* Select the Topomesh Control applet in the scrolling list
* Finalize your layout by right click and "Lock Layout"

Mesh objects stored as PropertyTopomesh structures can now be visualized simply by the command

```python
world.add(topomesh,"topomesh")
```

### VisuAlea components for mesh processing

Add functionalities handling PropertyTopomesh objects directly as visual progamming bricks

