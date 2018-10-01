# Extract Face Landmark Points 

Command line tool to extract landmarks in a face picture using OpenCV and DLib

Usage:

```bash
python3 extractLP.py <sourceglob>
```

Example:

```bash
python3 extractLP.py "**/*.jpg"
```


this will extract the points and will generate a JSON array into the **stdout**, so you can send it to any file/application for futher proccessing.