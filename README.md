# RemoveBackground
Remove Image Background with AI

Step by Step Guide:

1. Got to proper path for cloning the projct.
2. Clone the repository:

```bash
git clone https://github.com/karrabi/RemoveBackground.git
```
3. Go inside created folder
```bash
cd RemoveBackground
```
4. Create a python virtual environment for the project:
```bash
python -m venv venv
```
5. Activate the virtual environment
```bash
venv\Scripts\activate
```
6. Install packages
```bash
pip install -r requirements.txt
```

wait until all requirements installed

7. Run the API
```bash
python app.py
```

Now back-end server is running and waiting for front-end

8. Run front-end
    - Double-Click on ***remove-background.html*** file and open it in a web browser
    - Select an Image from your computer with *Select Image* button. The selected image will shows in **Original Image** frame
    - Click on **Process Image** button and wait for result. it may takes a few seconds based on your system resources.
    - The processed image will shows on *Processed Image* frame.


